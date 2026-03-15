def getRoot(config):
  if not config.parent:
    return config
  return getRoot(config.parent)

root = getRoot(config)

if root.host_os not in ['Linux']:
  config.unsupported = True
if root.target_arch not in ['x86_64']:
  config.unsupported = True
if root.support_amd_offload_tests == 'false':
  config.unsupported = True
