def getRoot(config):
    if not config.parent:
        return config
    return getRoot(config.parent)


root = getRoot(config)

if root.android != "1":
    config.unsupported = True

config.substitutions.append(("%device", "/data/local/tmp/Output"))
