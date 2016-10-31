def getRoot(config):
  if not config.parent:
    return config
  return getRoot(config.parent)

root = getRoot(config)

# Only run if we have an instrumented libcxx.  On Darwin, run always (we have
# interceptors to support the system-provided libcxx).
if not root.has_libcxx and root.host_os != 'Darwin':
  config.unsupported = True

