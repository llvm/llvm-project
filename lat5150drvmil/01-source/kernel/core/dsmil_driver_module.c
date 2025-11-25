/*
 * DSMIL Driver Module Wrapper
 *
 * Builds the primary DSMIL kernel module as a distinct translation unit so
 * that additional objects (for example the Rust safety layer) can be linked
 * in via Kbuild without renaming the original driver source file.
 */

#include "dsmil-104dev.c"
