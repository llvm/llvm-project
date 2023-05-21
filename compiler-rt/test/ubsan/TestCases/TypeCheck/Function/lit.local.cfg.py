if config.host_os not in ['Darwin', 'FreeBSD', 'Linux', 'NetBSD']:
  config.unsupported = True
# Work around "library ... not found: needed by main executable" in qemu.
if config.android and config.target_arch == 'arm':
  config.unsupported = True
