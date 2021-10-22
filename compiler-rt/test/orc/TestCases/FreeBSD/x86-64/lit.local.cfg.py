if config.root.host_arch not in ['x86_64', 'amd64']:
  config.unsupported = True

if config.target_arch not in ['x86_64', 'amd64']:
  config.unsupported = True