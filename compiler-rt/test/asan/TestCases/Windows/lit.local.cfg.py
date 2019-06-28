def getRoot(config):
  if not config.parent:
    return config
  return getRoot(config.parent)

root = getRoot(config)

# We only run a small set of tests on Windows for now.
# Override the parent directory's "unsupported" decision until we can handle
# all of its tests.
if root.host_os in ['Windows']:
  config.unsupported = False
else:
  config.unsupported = True
