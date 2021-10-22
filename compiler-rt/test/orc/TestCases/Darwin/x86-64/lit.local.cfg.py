if config.root.host_arch != 'x86_64':
  config.unsupported = True

if config.target_arch != 'x86_64':
  config.unsupported = True