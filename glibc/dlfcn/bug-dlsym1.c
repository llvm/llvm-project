/* Test case for bug in dlsym accessing dependency objects' symbols.  */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <dlfcn.h>

int main(void)
{
    void *handle;
    char *c;

    /* open lib1.so, which has the unresolved test symbol and a DT_NEEDED
       on lib2.so, which provides the symbol */
    if ((handle = dlopen("bug-dlsym1-lib1.so", RTLD_NOW)) == NULL) {
	printf("dlopen(\"bug-dlsym1-lib1.so\"): %s\n", dlerror());
	abort();
    }

    if ((c = dlsym(handle, "dlopen_test_variable")) == NULL) {
	printf("dlsym(handle, \"dlopen_test_variable\"): %s\n", dlerror());
	abort();
    }

    (void) dlclose(handle);

    return 0;
}
