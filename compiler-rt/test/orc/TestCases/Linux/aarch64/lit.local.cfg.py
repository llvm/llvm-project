if config.root.host_arch != 'aarch64':
  config.unsupported = True

if config.target_arch != 'aarch64':
  config.unsupported = True
