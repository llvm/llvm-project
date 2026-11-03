print(config.available_features)

if config.root.target_os != "Darwin":
    config.unsupported = True
