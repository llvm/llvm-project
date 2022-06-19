def getRoot(config):
  if not config.parent:
    return config
  return getRoot(config.parent)

root = getRoot(config)

if root.host_os in ['Windows']:
  config.unsupported = True

# AIX usually usually makes use of an explicit export list when linking a shared
# object, since the linker doesn't export anything by default.
if root.host_os in ['AIX']:
  config.substitutions.append(('%shared_linker_xopts', '-Wl,-bE:shr.exp'))
else:
  config.substitutions.append(('%shared_linker_xopts', ''))
