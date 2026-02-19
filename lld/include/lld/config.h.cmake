#ifndef CONFIG_H
#define CONFIG_H

// Include this header only under the lld source tree.
// This is a private header.

/* Allow LLD to link to GPLv3-licensed files. */
#cmakedefine01 LLD_LINK_GPL3

/* Enable support for GNU LTO Format, i.e. use LLD to link GCC LTO files. */
#cmakedefine01 LLD_ENABLE_GNU_LTO

#endif
