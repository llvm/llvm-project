extern void a_function (void);
extern void b_function (void);
extern void c_function (void);

void
a_function (void)
{
  b_function ();
  c_function ();
}
