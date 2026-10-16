def getRoot(config):
    if not config.parent:
        return config
    return getRoot(config.parent)


root = getRoot(config)

if root.target_os not in ["Linux", "FreeBSD", "NetBSD"]:
    config.unsupported = True

# Android O (API level 26) has support for cross-dso cfi in libdl.so.
if config.android and "android-26" not in config.available_features:
    config.unsupported = True

# The runtime library only supports 4K pages.
if "page-size-4096" not in config.available_features:
    config.unsupported = True
