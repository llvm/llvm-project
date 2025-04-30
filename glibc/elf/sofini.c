/* Terminate the frame unwind info section with a 4byte 0 as a sentinel;
   this would be the 'length' field in a real FDE.  */

typedef unsigned int ui32 __attribute__ ((mode (SI)));
static const ui32 __FRAME_END__[1]
  __attribute__ ((used, section (".eh_frame")))
  = { 0 };
