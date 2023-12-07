def getRoot(config):
    if not config.parent:
        return config
    return getRoot(config.parent)


root = getRoot(config)

if root.host_os not in ["Darwin"]:
    config.unsupported = True

config.environment["TSAN_OPTIONS"] += ":ignore_noninstrumented_modules=1"
