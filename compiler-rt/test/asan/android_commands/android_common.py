import os, subprocess, tempfile
import time

ANDROID_TMPDIR = '/data/local/tmp/Output'
ADB = os.environ.get('ADB', 'adb')

verbose = False
if os.environ.get('ANDROID_RUN_VERBOSE') == '1':
    verbose = True

def adb(args, attempts = 1):
    if verbose:
        print args
    tmpname = tempfile.mktemp()
    out = open(tmpname, 'w')
    ret = 255
    while attempts > 0 and ret != 0:
      attempts -= 1
      ret = subprocess.call([ADB] + args, stdout=out, stderr=subprocess.STDOUT)
      if attempts != 0:
        ret = 5
    if ret != 0:
      print "adb command failed", args
      print tmpname
      out.close()
      out = open(tmpname, 'r')
      print out.read()
    out.close()
    os.unlink(tmpname)
    return ret

def pull_from_device(path):
    tmp = tempfile.mktemp()
    adb(['pull', path, tmp], 5)
    text = open(tmp, 'r').read()
    os.unlink(tmp)
    return text

def push_to_device(path):
    # Workaround for https://code.google.com/p/android/issues/detail?id=65857
    dst_path = os.path.join(ANDROID_TMPDIR, os.path.basename(path))
    tmp_path = dst_path + '.push'
    adb(['push', path, tmp_path], 5)
    adb(['shell', 'cp "%s" "%s" 2>&1' % (tmp_path, dst_path)], 5)
