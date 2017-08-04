#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static void func_a (void) __attribute__((noinline));
static void func_b (void) __attribute__((noreturn));
static void func_c (void) __attribute__((noinline));

static void
func_c (void)
{
	abort ();
}

static void
func_b (void)
{
	func_c ();
	while (1)
        ;
}

static void
func_a (void)
{
	func_b ();
}

int
main (int argc, char *argv[])
{
    sleep (2);

	func_a ();

	return 0;
}
