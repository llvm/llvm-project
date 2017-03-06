// Check for GNUMode = 1 by looking for the "linux" define which only exists
// for GNUMode = 1.
#ifdef linux
 #error "Submodule has GNUMode=1"
#endif
