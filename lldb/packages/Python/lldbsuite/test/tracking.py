#!/usr/bin/env python
import os
import os.path
import datetime
import re
import sys
sys.path.append(os.path.join(os.getcwd(), 'radarclient-python'))


def usage():
    print "Run this from somewhere fancy"
    print "Install pycurl and radarclient"
    print "To install radarclient - while in the LLDB test repository do 'git clone https://your_od_username@swegit.apple.com/git/radarclient-python.git'"
    print "To install pycurl - from anywhere, do 'sudo easy_install pycurl'"
    sys.exit(1)

try:
    from radarclient import *
except:
    usage()


def find_session_dir():
    # Use heuristic to find the latest session directory.
    name = datetime.datetime.now().strftime("%Y-%m-%d-")
    dirs = [d for d in os.listdir(os.getcwd()) if d.startswith(name)]
    if len(dirs) == 0:
        print "No default session directory found, please specify it explicitly."
        usage()
    session_dir = max(dirs, key=os.path.getmtime)
    if not session_dir or not os.path.exists(session_dir):
        print "No default session directory found, please specify it explicitly."
        usage()
    return os.path.join(os.getcwd(), session_dir)


def process_no_bug(path, txt):
    print 'Failure %s has no bug tracking (says %s) - why?' % (path, txt)


def process_llvm_bug(path, pr):
    print 'Failure %s has an LLVM bug tracking: %s - why no radar?' % (path, pr.group(1))


def process_radar_bug(path, rdr):
    rdr = get_radar_details(rdr.group(1))
    if not rdr:
        print 'Failure %s claims to be tracked by rdar://%s but no such bug exists - consider filing one' % (path, rdr.group(1))
        return
    print 'Failure %s has a radar bug tracking: rdar://%s %s - cool' % (path, rdr.id, rdr.title)
    if rdr.state != 'Analyze':
        print 'This radar is not in Analyze anymore - Consider filing a new one, or editing the test case'
    print '  Assignee: %s %s' % (rdr.assignee.firstName, rdr.assignee.lastName)
    print '  Milestone: %s' % (rdr.milestone[u'name'] if rdr.milestone else 'None')
    print '  Priority: %s' % (rdr.priority)

global_radar_client = None


def get_radar_details(id):
    global global_radar_client
    if global_radar_client is None:
        authentication_strategy = AuthenticationStrategySPNego()
        system_identifier = ClientSystemIdentifier('lldb-test-tracker', '1.0')
        global_radar_client = RadarClient(
            authentication_strategy, system_identifier)
        global_radar_client.problem_default_fields = [
            'id',
            'title',
            'assignee',
            'milestone',
            'component',
            'priority',
            'fixOrder',
            'state',
            'substate',
            'resolution',
            'duplicateOfProblemID']
    rdar = global_radar_client.radar_for_id(id)
    if rdar.state == 'Verify' or rdar.state == 'Closed' and rdar.resolution == 'Duplicate':
        return get_radar_details(rdar.duplicateOfProblemID)
    return rdar


def process_xfail(path):
    marker = 'expected failure (problem id:'
    content = ""
    with open(path, 'r') as content_file:
        content = content_file.read()
    name = os.path.basename(path)
    try:
        name = name[name.find('-') + 1:]
        name = name[name.find('-') + 1:]
        name = name[name.find('-') + 1:]
        name = name.replace('.log', '')
        name = name[:name.rfind('.') - 1]
    finally:
        pass
    xfail_line = content[content.find(
        'expected failure (problem id:') + len(marker):]
    xfail_line = xfail_line[:xfail_line.find('\n')]
    m1 = re.search('rdar://([0-9]+)', xfail_line)
    m2 = re.search('rdar://problem/([0-9]+)', xfail_line)
    m3 = re.search('llvm.org/pr([0-9]+)', xfail_line)
    if m1 is None and m2 is None:
        if m3 is None:
            process_no_bug(name, xfail_line)
        else:
            process_llvm_bug(name, m3)
    else:
        process_radar_bug(name, m1 if m1 else m2)

    print ""

session_dir_path = find_session_dir()
import os
for root, dirs, files in os.walk(session_dir_path, topdown=False):
    for name in files:
        if name.startswith("ExpectedFailure"):
            process_xfail(os.path.join(session_dir_path, name))
