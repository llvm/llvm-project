# TODO: jitlink for ppc64/powerpc64 hasn't been well tested yet.
# We should support it in the future.
if config.root.host_arch != "ppc64le":
    config.unsupported = True

if config.target_arch != "powerpc64le":
    config.unsupported = True
