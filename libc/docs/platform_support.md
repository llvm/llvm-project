# Platform Support

Development is currently mostly focused on Linux. MacOS and Windows has
partial support, but has bitrot and isn't being tested continuously.

LLVM-libc is currently being integrated into Android and Fuchsia operating
systems via {doc}`overlay mode <overlay_mode>`.

For Linux, we support kernel versions as listed on
[kernel.org](https://kernel.org/), including `longterm` (not past EOL
date), `stable`, and `mainline` versions. We actively adopt new features
from `linux-next`.

For Windows, we plan to support products within their lifecycle. Please refer to
[Search Product and Services Lifecycle Information](https://learn.microsoft.com/en-us/lifecycle/products/?products=windows) for more information.

LLVM-libc does not guarantee backward compatibility with operating systems that
have reached their EOL. Compatibility patches for obsolete operating systems
will not be accepted.

For GPU, reference {doc}`our GPU docs <gpu/index>`.
