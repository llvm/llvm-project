import subprocess

def getRoot(config):
  if not config.parent:
    return config
  return getRoot(config.parent)

root = getRoot(config)

# As this has not been tested extensively on non-Darwin platforms,
# only Darwin support is enabled for the moment. However, continuous mode
# may "just work" without modification on Linux and other UNIX-likes (AIUI
# the default value for the GNU linker's `--section-alignment` flag is
# 0x1000, which is the size of a page on many systems).
#
# Please add supported configs to this list.
if root.host_os not in ['Darwin']:
  config.unsupported = True
