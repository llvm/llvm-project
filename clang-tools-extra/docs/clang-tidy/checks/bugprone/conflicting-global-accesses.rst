.. title:: clang-tidy - bugprone-conflicting-global-accesses

bugprone-conflicting-global-accesses
====================================

Finds conflicting accesses on global variables.

Modifying twice or reading and modifying a memory location without a
defined sequence of the operations is either undefined behavior or has
unspecified order. This checker is similar to the -Wunsequenced clang warning,
however it only looks at global variables and therefore can find conflicting
actions recursively inside functions as well.

For example::

    int a = 0;
    int b = (a++) - a; // This is flagged by -Wunsequenced.

However global variables allow for more complex scenarios that
-Wunsequenced doesn't detect. E.g. ::

    int globalVar = 0;

    int incFun() {
      globalVar++;
      return globalVar;
    }

    int main() {
      return globalVar + incFun(); // This is not detected by -Wunsequenced.
    }

This checker attempts to detect such cases. It recurses into functions that are
inside the same translation unit. It also attempts not to flag cases that are
already covered by -Wunsequenced. Global unions and structs are also handled.
For example::
    
    typedef struct {
        int A;
        float B;
    } IntAndFloat;

    IntAndFloat GlobalIF;

    int globalIFGetSum() {
        int sum = GlobalIF.A + (int)GlobalIF.B;
        GlobalIF = (IntAndFloat){};
        return sum;
    }

    int main() {
        // The following printf could give different results on different
        // compilers.
        printf("sum: %i, int: %i", globalIFGetSum(), GlobalIF.A);
    }

Options
-------

.. option:: HandleMutableFunctionParametersAsWrites

    It's possible to enable handling mutable reference and pointer function
    parameters as writes using the HandleMutableFunctionParametersAsWrites
    option. For example:

    void func(int& a);

    int globalVar;
    func(globalVar); // <- this could be a write to globalVar.

    This option is disabled by default.

