/* A simple valid linker script used for testing the -T/--script options.
 *
 * An unresolved symbol named '_entry_point' can be scanned for by the tests
 * to determine that the linker script was processed.
 */
ENTRY(_entry_point)
