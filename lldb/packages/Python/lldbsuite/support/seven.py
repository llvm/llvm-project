import binascii
import six
import shlex

if six.PY2:
    import commands
    get_command_output = commands.getoutput
    get_command_status_output = commands.getstatusoutput

    cmp_ = cmp
else:
    def get_command_status_output(command):
        try:
            import subprocess
            return (
                0,
                subprocess.check_output(
                    command,
                    shell=True,
                    universal_newlines=True).rstrip())
        except subprocess.CalledProcessError as e:
            return (e.returncode, e.output)

    def get_command_output(command):
        return get_command_status_output(command)[1]

    cmp_ = lambda x, y: (x > y) - (x < y)

def bitcast_to_string(b):
    """
    Take a string(PY2) or a bytes(PY3) object and return a string. The returned
    string contains the exact same bytes as the input object (latin1 <-> unicode
    transformation is an identity operation for the first 256 code points).
    """
    return b if six.PY2 else b.decode("latin1")

def bitcast_to_bytes(s):
    """
    Take a string and return a string(PY2) or a bytes(PY3) object. The returned
    object contains the exact same bytes as the input string. (latin1 <->
    unicode transformation is an identity operation for the first 256 code
    points).
    """
    return s if six.PY2 else s.encode("latin1")

def unhexlify(hexstr):
    """Hex-decode a string. The result is always a string."""
    return bitcast_to_string(binascii.unhexlify(hexstr))

def hexlify(data):
    """Hex-encode string data. The result if always a string."""
    return bitcast_to_string(binascii.hexlify(bitcast_to_bytes(data)))

# TODO: Replace this with `shlex.join` when minimum Python version is >= 3.8
def join_for_shell(split_command):
    return " ".join([shlex.quote(part) for part in split_command])
