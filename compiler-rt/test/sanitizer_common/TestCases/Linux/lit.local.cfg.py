def getRoot(config):
  if not config.parent:
    return config
  return getRoot(config.parent)

root = getRoot(config)

if root.host_os not in ['Linux']:
  config.unsupported = True
# FIXME https://github.com/llvm/llvm-project/issues/54084
if root.host_arch in ['ppc64', 'ppc64le']:
  config.unsupported = True
