if config.root.host_arch not in ['aarch64', 'arm64']:
  config.unsupported = True

if config.target_arch not in ['aarch64', 'arm64']:
  config.unsupported = True