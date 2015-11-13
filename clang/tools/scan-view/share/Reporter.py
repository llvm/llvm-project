"""Methods for reporting bugs."""

import subprocess, sys, os

__all__ = ['ReportFailure', 'BugReport', 'getReporters']

#

class ReportFailure(Exception):
    """Generic exception for failures in bug reporting."""
    def __init__(self, value):        
        self.value = value

# Collect information about a bug.

class BugReport:
    def __init__(self, title, description, files):
        self.title = title
        self.description = description
        self.files = files

# Reporter interfaces.

import os

import email, mimetypes, smtplib
from email import encoders
from email.message import Message
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

#===------------------------------------------------------------------------===#
# ReporterParameter
#===------------------------------------------------------------------------===#

class ReporterParameter:
  def __init__(self, n):
    self.name = n
  def getName(self):
    return self.name
  def getValue(self,r,bugtype,getConfigOption):
     return getConfigOption(r.getName(),self.getName())
  def saveConfigValue(self):
    return True

class TextParameter (ReporterParameter):
  def getHTML(self,r,bugtype,getConfigOption):
    return """\
<tr>
<td class="form_clabel">%s:</td>
<td class="form_value"><input type="text" name="%s_%s" value="%s"></td>
</tr>"""%(self.getName(),r.getName(),self.getName(),self.getValue(r,bugtype,getConfigOption))

class SelectionParameter (ReporterParameter):
  def __init__(self, n, values):
    ReporterParameter.__init__(self,n)
    self.values = values
    
  def getHTML(self,r,bugtype,getConfigOption):
    default = self.getValue(r,bugtype,getConfigOption)
    return """\
<tr>
<td class="form_clabel">%s:</td><td class="form_value"><select name="%s_%s">
%s
</select></td>"""%(self.getName(),r.getName(),self.getName(),'\n'.join(["""\
<option value="%s"%s>%s</option>"""%(o[0],
                                     o[0] == default and ' selected="selected"' or '',
                                     o[1]) for o in self.values]))

#===------------------------------------------------------------------------===#
# Reporters
#===------------------------------------------------------------------------===#

class EmailReporter:
    def getName(self):
        return 'Email'

    def getParameters(self):
        return map(lambda x:TextParameter(x),['To', 'From', 'SMTP Server', 'SMTP Port'])

    # Lifted from python email module examples.
    def attachFile(self, outer, path):
        # Guess the content type based on the file's extension.  Encoding
        # will be ignored, although we should check for simple things like
        # gzip'd or compressed files.
        ctype, encoding = mimetypes.guess_type(path)
        if ctype is None or encoding is not None:
            # No guess could be made, or the file is encoded (compressed), so
            # use a generic bag-of-bits type.
            ctype = 'application/octet-stream'
        maintype, subtype = ctype.split('/', 1)
        if maintype == 'text':
            fp = open(path)
            # Note: we should handle calculating the charset
            msg = MIMEText(fp.read(), _subtype=subtype)
            fp.close()
        else:
            fp = open(path, 'rb')
            msg = MIMEBase(maintype, subtype)
            msg.set_payload(fp.read())
            fp.close()
            # Encode the payload using Base64
            encoders.encode_base64(msg)
        # Set the filename parameter
        msg.add_header('Content-Disposition', 'attachment', filename=os.path.basename(path))
        outer.attach(msg)

    def fileReport(self, report, parameters):
        mainMsg = """\
BUG REPORT
---
Title: %s
Description: %s
"""%(report.title, report.description)

        if not parameters.get('To'):
            raise ReportFailure('No "To" address specified.')
        if not parameters.get('From'):
            raise ReportFailure('No "From" address specified.')

        msg = MIMEMultipart()
        msg['Subject'] = 'BUG REPORT: %s'%(report.title)
        # FIXME: Get config parameters
        msg['To'] = parameters.get('To')
        msg['From'] = parameters.get('From')
        msg.preamble = mainMsg

        msg.attach(MIMEText(mainMsg, _subtype='text/plain'))
        for file in report.files:
            self.attachFile(msg, file)

        try:
            s = smtplib.SMTP(host=parameters.get('SMTP Server'),
                             port=parameters.get('SMTP Port'))
            s.sendmail(msg['From'], msg['To'], msg.as_string())
            s.close()
        except:
            raise ReportFailure('Unable to send message via SMTP.')

        return "Message sent!"

class BugzillaReporter:
    def getName(self):
        return 'Bugzilla'
    
    def getParameters(self):
        return map(lambda x:TextParameter(x),['URL','Product'])

    def fileReport(self, report, parameters):
        raise NotImplementedError
 

class RadarClassificationParameter(SelectionParameter):
  def __init__(self):
    SelectionParameter.__init__(self,"Classification",
            [['1', 'Security'], ['2', 'Crash/Hang/Data Loss'],
             ['3', 'Performance'], ['4', 'UI/Usability'], 
             ['6', 'Serious Bug'], ['7', 'Other']])

  def saveConfigValue(self):
    return False
    
  def getValue(self,r,bugtype,getConfigOption):
    if bugtype.find("leak") != -1:
      return '3'
    elif bugtype.find("dereference") != -1:
      return '2'
    elif bugtype.find("missing ivar release") != -1:
      return '3'
    else:
      return '7'

class RadarReporter:
    @staticmethod
    def isAvailable():
        # FIXME: Find this .scpt better
        path = os.path.join(os.path.dirname(__file__),'../share/scan-view/GetRadarVersion.scpt')
        try:
          p = subprocess.Popen(['osascript',path], 
          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except:
            return False
        data,err = p.communicate()
        res = p.wait()
        # FIXME: Check version? Check for no errors?
        return res == 0

    def getName(self):
        return 'Radar'

    def getParameters(self):
        return [ TextParameter('Component'), TextParameter('Component Version'),
                 RadarClassificationParameter() ]

    def fileReport(self, report, parameters):
        component = parameters.get('Component', '')
        componentVersion = parameters.get('Component Version', '')
        classification = parameters.get('Classification', '')
        personID = ""
        diagnosis = ""
        config = ""

        if not component.strip():
            component = 'Bugs found by clang Analyzer'
        if not componentVersion.strip():
            componentVersion = 'X'

        script = os.path.join(os.path.dirname(__file__),'../share/scan-view/FileRadar.scpt')
        args = ['osascript', script, component, componentVersion, classification, personID, report.title,
                report.description, diagnosis, config] + map(os.path.abspath, report.files)
#        print >>sys.stderr, args
        try:
          p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except:
            raise ReportFailure("Unable to file radar (AppleScript failure).")
        data, err = p.communicate()
        res = p.wait()

        if res:
            raise ReportFailure("Unable to file radar (AppleScript failure).")

        try:
            values = eval(data)
        except:
            raise ReportFailure("Unable to process radar results.")

        # We expect (int: bugID, str: message)
        if len(values) != 2 or not isinstance(values[0], int):
            raise ReportFailure("Unable to process radar results.")

        bugID,message = values
        bugID = int(bugID)
        
        if not bugID:
            raise ReportFailure(message)
        
        return "Filed: <a href=\"rdar://%d/\">%d</a>"%(bugID,bugID)

###

def getReporters():
    reporters = []
    if RadarReporter.isAvailable():
        reporters.append(RadarReporter())
    reporters.append(EmailReporter())
    return reporters

