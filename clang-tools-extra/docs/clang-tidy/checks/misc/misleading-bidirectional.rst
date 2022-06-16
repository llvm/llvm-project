.. title:: clang-tidy - misc-misleading-bidirectional

misc-misleading-bidirectional
=============================

Warn about unterminated bidirectional unicode sequence, detecting potential attack
as described in the `Trojan Source <https://www.trojansource.codes>`_ attack.

Example:

.. code-block:: c++

    #include <iostream>

    int main() {
        bool isAdmin = false;
        /*‮ } ⁦if (isAdmin)⁩ ⁦ begin admins only */
            std::cout << "You are an admin.\n";
        /* end admins only ‮ { ⁦*/
        return 0;
    }
