extern void a_function (void);
extern void b_function (void);
extern void c_function (void);
extern void d_function (void);

void
d_function (void)
{
  a_function ();
  b_function ();
  c_function ();
}
