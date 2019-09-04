These tests concern the experimental Swift feature `@_implementationOnly import`.\* Unlike a normal import, an `@_implementationOnly` import is guaranteed not to be part of a library's API or ABI, which means that client apps don't have to know about the import at all to use the library. However, the import *can* be used in the *implementation* of the library, so it's important for LLDB to handle the cases where the `@_implementationOnly` import is and isn't available when debugging.

In a scenario with a client app, a library, and an implementation-only import that the library uses, there are three situations we care about:

1. The library has no debug info, or just line tables. In this case, the implementation-only import is irrelevant; someone debugging the app can only use the public parts of the library.

2. The library has debug info and the implementation-only import is available. In this case, the debugger should make everything available like it would with a normal import.

3. The library has debug info, but the implementation-only import is not available (for whatever reason). In this case LLDB may still have to deal with internal parts of the library even though some types will not be available. (This is the least important case, but it'd be good to not crash or lie to users.)

\* Hopefully someone will remember to edit this ReadMe before the feature goes public!
