// RUN: %clang_cc1 -triple s390x-linux-gnu -fzvector \
// RUN:  -flax-vector-conversions=none -W -Wall -Wconversion \
// RUN:  -Werror -fsyntax-only -verify %s

vector signed char sc, sc2;
vector unsigned char uc, uc2;
vector bool char bc, bc2;

vector signed short ss, ss2;
vector unsigned short us, us2;
vector bool short bs, bs2;

vector signed int si, si2;
vector unsigned int ui, ui2;
vector bool int bi, bi2;

vector signed long long sl, sl2;
vector unsigned long long ul, ul2;
vector bool long long bl, bl2;

vector signed __int128 slll, slll2;
vector unsigned __int128 ulll, ulll2;
vector bool __int128 blll, blll2;

vector double fd, fd2;

vector long ll; // expected-error {{cannot use 'long' with '__vector'}}
vector float ff; // expected-error {{cannot use 'float' with '__vector'}}
vector long double ld; // expected-error {{cannot use 'long double' with '__vector'}}
vector float _Complex cf; // expected-error {{cannot use '_Complex' with '__vector'}}
vector double _Complex cd; // expected-error {{cannot use '_Complex' with '__vector'}}
vector long double _Complex cld; // expected-error {{cannot use '_Complex' with '__vector'}}

signed char sc_scalar;
unsigned char uc_scalar;

signed short ss_scalar;
unsigned short us_scalar;

signed int si_scalar;
unsigned int ui_scalar;

signed long sl_scalar;
unsigned long ul_scalar;

signed __int128 slll_scalar;
unsigned __int128 ulll_scalar;

double fd_scalar;

// Verify that __vector is also recognized
__vector signed char sc3;
__vector unsigned char uc3;
__vector bool char bc3;
__vector signed short ss3;
__vector unsigned short us3;
__vector bool short bs3;
__vector signed int si3;
__vector unsigned int ui3;
__vector bool int bi3;
__vector signed long long sl3;
__vector unsigned long long ul3;
__vector bool long long bl3;
__vector signed __int128 slll3;
__vector unsigned __int128 ulll3;
__vector bool __int128 blll3;
__vector double fd3;
__vector long ll3; // expected-error {{cannot use 'long' with '__vector'}}
__vector float ff3; // expected-error {{cannot use 'float' with '__vector'}}
__vector long double ld3; // expected-error {{cannot use 'long double' with '__vector'}}
__vector float _Complex cf3; // expected-error {{cannot use '_Complex' with '__vector'}}
__vector double _Complex cd3; // expected-error {{cannot use '_Complex' with '__vector'}}
__vector long double _Complex cld3; // expected-error {{cannot use '_Complex' with '__vector'}}

// Likewise for __bool
vector __bool char bc4;
vector __bool short bs4;
vector __bool int bi4;
vector __bool long long bl4;
__vector __bool char bc5;
__vector __bool short bs5;
__vector __bool int bi5;
__vector __bool long long bl5;

// Verify operation of vec_step
int res_sc[vec_step(sc) == 16 ? 1 : -1];
int res_uc[vec_step(uc) == 16 ? 1 : -1];
int res_bc[vec_step(bc) == 16 ? 1 : -1];
int res_ss[vec_step(ss) == 8 ? 1 : -1];
int res_us[vec_step(us) == 8 ? 1 : -1];
int res_bs[vec_step(bs) == 8 ? 1 : -1];
int res_si[vec_step(si) == 4 ? 1 : -1];
int res_ui[vec_step(ui) == 4 ? 1 : -1];
int res_bi[vec_step(bi) == 4 ? 1 : -1];
int res_sl[vec_step(sl) == 2 ? 1 : -1];
int res_ul[vec_step(ul) == 2 ? 1 : -1];
int res_bl[vec_step(bl) == 2 ? 1 : -1];
int res_slll[vec_step(slll) == 1 ? 1 : -1];
int res_ulll[vec_step(ulll) == 1 ? 1 : -1];
int res_blll[vec_step(blll) == 1 ? 1 : -1];
int res_fd[vec_step(fd) == 2 ? 1 : -1];


void foo(void)
{
  // -------------------------------------------------------------------------
  // Test assignment.
  // -------------------------------------------------------------------------

  sc = sc2;
  uc = uc2;
  bc = bc2;

  ss = ss2;
  us = us2;
  bs = bs2;

  si = si2;
  ui = ui2;
  bi = bi2;

  sl = sl2;
  ul = ul2;
  bl = bl2;
  fd = fd2;

  slll = slll2;
  ulll = ulll2;
  blll = blll2;

  sc = uc2; // expected-error {{incompatible type}}
  sc = bc2; // expected-error {{incompatible type}}
  uc = sc2; // expected-error {{incompatible type}}
  uc = bc2; // expected-error {{incompatible type}}
  bc = sc2; // expected-error {{incompatible type}}
  bc = uc2; // expected-error {{incompatible type}}

  sc = sc_scalar; // expected-error {{incompatible type}}
  sc = uc_scalar; // expected-error {{incompatible type}}
  uc = sc_scalar; // expected-error {{incompatible type}}
  uc = uc_scalar; // expected-error {{incompatible type}}
  bc = sc_scalar; // expected-error {{incompatible type}}
  bc = uc_scalar; // expected-error {{incompatible type}}

  sc = ss2; // expected-error {{incompatible type}}
  sc = si2; // expected-error {{incompatible type}}
  sc = sl2; // expected-error {{incompatible type}}
  sc = fd2; // expected-error {{incompatible type}}
  sc = slll2; // expected-error {{incompatible type}}

  ss = sc2; // expected-error {{incompatible type}}
  si = sc2; // expected-error {{incompatible type}}
  sl = sc2; // expected-error {{incompatible type}}
  fd = sc2; // expected-error {{incompatible type}}
  slll = sc2; // expected-error {{incompatible type}}

  uc = us2; // expected-error {{incompatible type}}
  uc = ui2; // expected-error {{incompatible type}}
  uc = ul2; // expected-error {{incompatible type}}
  uc = fd2; // expected-error {{incompatible type}}
  uc = ulll2; // expected-error {{incompatible type}}

  us = uc2; // expected-error {{incompatible type}}
  ui = uc2; // expected-error {{incompatible type}}
  ul = uc2; // expected-error {{incompatible type}}
  fd = uc2; // expected-error {{incompatible type}}
  ulll = uc2; // expected-error {{incompatible type}}

  bc = us2; // expected-error {{incompatible type}}
  bc = ui2; // expected-error {{incompatible type}}
  bc = ul2; // expected-error {{incompatible type}}
  bc = fd2; // expected-error {{incompatible type}}
  bc = ulll2; // expected-error {{incompatible type}}

  bs = bc2; // expected-error {{incompatible type}}
  bi = bc2; // expected-error {{incompatible type}}
  bl = bc2; // expected-error {{incompatible type}}
  fd = bc2; // expected-error {{incompatible type}}
  blll = bc2; // expected-error {{incompatible type}}

  // -------------------------------------------------------------------------
  // Test casts to same element width.
  // -------------------------------------------------------------------------

  sc = (vector signed char)bc2;
  bc = (vector bool char)uc2;
  uc = (vector unsigned char)sc2;

  ss = (vector signed short)bs2;
  bs = (vector bool short)us2;
  us = (vector unsigned short)ss2;

  si = (vector signed int)bi2;
  bi = (vector bool int)ui2;
  ui = (vector unsigned int)si2;

  sl = (vector signed long long)bl2;
  bl = (vector bool long long)ul2;
  ul = (vector unsigned long long)fd2;
  fd = (vector double)sl2;

  slll = (vector signed __int128)blll2;
  blll = (vector bool __int128)ulll2;
  ulll = (vector unsigned __int128)slll2;

  // -------------------------------------------------------------------------
  // Test casts to different element width.
  // -------------------------------------------------------------------------

  sc = (vector signed char)bs2;
  bc = (vector bool char)us2;
  uc = (vector unsigned char)fd2;

  ss = (vector signed short)bi2;
  bs = (vector bool short)ui2;
  us = (vector unsigned short)fd2;

  si = (vector signed int)bl2;
  bi = (vector bool int)ul2;
  ui = (vector unsigned int)fd2;

  sl = (vector signed long long)bc2;
  bl = (vector bool long long)uc2;
  ul = (vector unsigned long long)sc2;
  fd = (vector double)sc2;

  slll = (vector signed __int128)bi2;
  blll = (vector bool __int128)ui2;
  ulll = (vector unsigned __int128)si2;

  // -------------------------------------------------------------------------
  // Test ++.
  // -------------------------------------------------------------------------

  ++sc2;
  ++uc2;
  ++bc2; // expected-error {{cannot increment}}

  ++ss2;
  ++us2;
  ++bs2; // expected-error {{cannot increment}}

  ++si2;
  ++ui2;
  ++bi2; // expected-error {{cannot increment}}

  ++sl2;
  ++ul2;
  ++bl2; // expected-error {{cannot increment}}

  ++slll2;
  ++ulll2;
  ++blll2; // expected-error {{cannot increment}}

  ++fd2;

  sc++;
  uc++;
  bc++; // expected-error {{cannot increment}}

  ss++;
  us++;
  bs++; // expected-error {{cannot increment}}

  si++;
  ui++;
  bi++; // expected-error {{cannot increment}}

  sl++;
  ul++;
  bl++; // expected-error {{cannot increment}}

  slll++;
  ulll++;
  blll++; // expected-error {{cannot increment}}

  fd++;

  // -------------------------------------------------------------------------
  // Test --.
  // -------------------------------------------------------------------------

  --sc2;
  --uc2;
  --bc2; // expected-error {{cannot decrement}}

  --ss2;
  --us2;
  --bs2; // expected-error {{cannot decrement}}

  --si2;
  --ui2;
  --bi2; // expected-error {{cannot decrement}}

  --sl2;
  --ul2;
  --bl2; // expected-error {{cannot decrement}}

  --slll2;
  --ulll2;
  --blll2; // expected-error {{cannot decrement}}

  --fd2;

  sc--;
  uc--;
  bc--; // expected-error {{cannot decrement}}

  ss--;
  us--;
  bs--; // expected-error {{cannot decrement}}

  si--;
  ui--;
  bi--; // expected-error {{cannot decrement}}

  sl--;
  ul--;
  bl--; // expected-error {{cannot decrement}}

  slll--;
  ulll--;
  blll--; // expected-error {{cannot decrement}}

  fd--;

  // -------------------------------------------------------------------------
  // Test unary +.
  // -------------------------------------------------------------------------

  sc = +sc2;
  uc = +uc2;
  bc = +bc2; // expected-error {{invalid argument type}}

  ss = +ss2;
  us = +us2;
  bs = +bs2; // expected-error {{invalid argument type}}

  si = +si2;
  ui = +ui2;
  bi = +bi2; // expected-error {{invalid argument type}}

  sl = +sl2;
  ul = +ul2;
  bl = +bl2; // expected-error {{invalid argument type}}

  slll = +slll2;
  ulll = +ulll2;
  blll = +blll2; // expected-error {{invalid argument type}}

  fd = +fd2;

  sc = +si2; // expected-error {{assigning to}}
  ui = +si2; // expected-error {{assigning to}}
  ui = +bi2; // expected-error {{invalid argument type}}

  // -------------------------------------------------------------------------
  // Test unary -.
  // -------------------------------------------------------------------------

  sc = -sc2;
  uc = -uc2;
  bc = -bc2; // expected-error {{invalid argument type}}

  ss = -ss2;
  us = -us2;
  bs = -bs2; // expected-error {{invalid argument type}}

  si = -si2;
  ui = -ui2;
  bi = -bi2; // expected-error {{invalid argument type}}

  sl = -sl2;
  ul = -ul2;
  bl = -bl2; // expected-error {{invalid argument type}}

  slll = -slll2;
  ulll = -ulll2;
  blll = -blll2; // expected-error {{invalid argument type}}

  fd = -fd2;

  sc = -si2; // expected-error {{assigning to}}
  ui = -si2; // expected-error {{assigning to}}
  ui = -bi2; // expected-error {{invalid argument type}}

  // -------------------------------------------------------------------------
  // Test ~.
  // -------------------------------------------------------------------------

  sc = ~sc2;
  uc = ~uc2;
  bc = ~bc2;

  ss = ~ss2;
  us = ~us2;
  bs = ~bs2;

  si = ~si2;
  ui = ~ui2;
  bi = ~bi2;

  sl = ~sl2;
  ul = ~ul2;
  bl = ~bl2;

  slll = ~slll2;
  ulll = ~ulll2;
  blll = ~blll2;

  fd = ~fd2; // expected-error {{invalid argument}}

  sc = ~si2; // expected-error {{assigning to}}
  ui = ~si2; // expected-error {{assigning to}}
  ui = ~bi2; // expected-error {{assigning to}}

  // -------------------------------------------------------------------------
  // Test binary +.
  // -------------------------------------------------------------------------

  sc = sc + sc2;
  sc = sc + uc2; // expected-error {{cannot convert}}
  sc = uc + sc2; // expected-error {{cannot convert}}
  sc = sc + bc2;
  sc = bc + sc2;

  uc = uc + uc2;
  uc = sc + uc2; // expected-error {{cannot convert}}
  uc = uc + sc2; // expected-error {{cannot convert}}
  uc = bc + uc2;
  uc = uc + bc2;

  bc = bc + bc2; // expected-error {{invalid operands}}
  bc = bc + uc2; // expected-error {{incompatible type}}
  bc = uc + bc2; // expected-error {{incompatible type}}
  bc = bc + sc2; // expected-error {{incompatible type}}
  bc = sc + bc2; // expected-error {{incompatible type}}

  sc = sc + sc_scalar;
  sc = sc + uc_scalar; // expected-error {{cannot convert between scalar type 'unsigned char' and vector type '__vector signed char' (vector of 16 'signed char' values) as implicit conversion would cause truncation}}
  sc = sc_scalar + sc;
  sc = uc_scalar + sc; // expected-error {{cannot convert between scalar type 'unsigned char' and vector type '__vector signed char' (vector of 16 'signed char' values) as implicit conversion would cause truncation}}
  uc = uc + sc_scalar; // expected-error {{implicit conversion changes signedness: 'signed char' to '__vector unsigned char' (vector of 16 'unsigned char' values)}}
  uc = uc + uc_scalar;
  uc = sc_scalar + uc; // expected-error {{implicit conversion changes signedness: 'signed char' to '__vector unsigned char' (vector of 16 'unsigned char' values)}}
  uc = uc_scalar + uc;

  ss = ss + ss2;
  us = us + us2;
  bs = bs + bs2; // expected-error {{invalid operands}}

  si = si + si2;
  ui = ui + ui2;
  bi = bi + bi2; // expected-error {{invalid operands}}

  sl = sl + sl2;
  ul = ul + ul2;
  bl = bl + bl2; // expected-error {{invalid operands}}

  slll = slll + slll2;
  ulll = ulll + ulll2;
  blll = blll + blll2; // expected-error {{invalid operands}}

  fd = fd + fd2;
  fd = fd + ul2; // expected-error {{cannot convert}}
  fd = sl + fd2; // expected-error {{cannot convert}}

  sc += sc2;
  sc += uc2; // expected-error {{cannot convert}}
  sc += bc2;

  uc += uc2;
  uc += sc2; // expected-error {{cannot convert}}
  uc += bc2;

  bc += bc2; // expected-error {{invalid operands}}
  bc += sc2; // expected-error {{cannot convert}}
  bc += uc2; // expected-error {{cannot convert}}

  sc += ss2; // expected-error {{cannot convert}}
  sc += si2; // expected-error {{cannot convert}}
  sc += sl2; // expected-error {{cannot convert}}
  sc += fd2; // expected-error {{cannot convert}}
  sc += slll2; // expected-error {{cannot convert}}

  sc += sc_scalar;
  sc += uc_scalar; // expected-error {{cannot convert between scalar type 'unsigned char' and vector type '__vector signed char' (vector of 16 'signed char' values) as implicit conversion would cause truncation}}
  uc += sc_scalar; // expected-error {{implicit conversion changes signedness: 'signed char' to '__vector unsigned char' (vector of 16 'unsigned char' values)}}
  uc += uc_scalar;

  ss += ss2;
  us += us2;
  bs += bs2; // expected-error {{invalid operands}}

  si += si2;
  ui += ui2;
  bi += bi2; // expected-error {{invalid operands}}

  sl += sl2;
  ul += ul2;
  bl += bl2; // expected-error {{invalid operands}}

  slll += slll2;
  ulll += ulll2;
  blll += blll2; // expected-error {{invalid operands}}

  fd += fd2;

  // -------------------------------------------------------------------------
  // Test that binary + rules apply to binary - too.
  // -------------------------------------------------------------------------

  sc = sc - sc2;
  uc = uc - uc2;
  bc = bc - bc2; // expected-error {{invalid operands}}

  sc = uc - sc2; // expected-error {{cannot convert}}
  sc = sc - bc2;
  uc = bc - uc2;

  sc -= sc2;
  uc -= uc2;
  bc -= bc2; // expected-error {{invalid operands}}

  sc -= uc2; // expected-error {{cannot convert}}
  uc -= bc2;
  bc -= sc2; // expected-error {{cannot convert}}

  ss -= ss2;
  us -= us2;
  bs -= bs2; // expected-error {{invalid operands}}

  si -= si2;
  ui -= ui2;
  bi -= bi2; // expected-error {{invalid operands}}

  sl -= sl2;
  ul -= ul2;
  bl -= bl2; // expected-error {{invalid operands}}

  slll -= slll2;
  ulll -= ulll2;
  blll -= blll2; // expected-error {{invalid operands}}

  fd -= fd2;

  // -------------------------------------------------------------------------
  // Test that binary + rules apply to * too.  64-bit integer multiplication
  // is not required by the spec and so isn't tested here.
  // -------------------------------------------------------------------------

  sc = sc * sc2;
  uc = uc * uc2;
  bc = bc * bc2; // expected-error {{invalid operands}}

  sc = uc * sc2; // expected-error {{cannot convert}}
  sc = sc * bc2; // expected-error {{cannot convert}}
  uc = bc * uc2; // expected-error {{cannot convert}}

  sc *= sc2;
  uc *= uc2;
  bc *= bc2; // expected-error {{invalid operands}}

  sc *= uc2; // expected-error {{cannot convert}}
  uc *= bc2; // expected-error {{cannot convert}}
  bc *= sc2; // expected-error {{cannot convert}}

  ss *= ss2;
  us *= us2;
  bs *= bs2; // expected-error {{invalid operands}}

  si *= si2;
  ui *= ui2;
  bi *= bi2; // expected-error {{invalid operands}}

  sl *= sl2;
  ul *= ul2;
  bl *= bl2; // expected-error {{invalid operands}}

  slll *= slll2;
  ulll *= ulll2;
  blll *= blll2; // expected-error {{invalid operands}}


  fd *= fd2;

  // -------------------------------------------------------------------------
  // Test that * rules apply to / too.
  // -------------------------------------------------------------------------

  sc = sc / sc2;
  uc = uc / uc2;
  bc = bc / bc2; // expected-error {{invalid operands}}

  sc = uc / sc2; // expected-error {{cannot convert}}
  sc = sc / bc2; // expected-error {{cannot convert}}
  uc = bc / uc2; // expected-error {{cannot convert}}

  sc /= sc2;
  uc /= uc2;
  bc /= bc2; // expected-error {{invalid operands}}

  sc /= uc2; // expected-error {{cannot convert}}
  uc /= bc2; // expected-error {{cannot convert}}
  bc /= sc2; // expected-error {{cannot convert}}

  ss /= ss2;
  us /= us2;
  bs /= bs2; // expected-error {{invalid operands}}

  si /= si2;
  ui /= ui2;
  bi /= bi2; // expected-error {{invalid operands}}

  sl /= sl2;
  ul /= ul2;
  bl /= bl2; // expected-error {{invalid operands}}

  slll /= slll2;
  ulll /= ulll2;
  blll /= blll2; // expected-error {{invalid operands}}

  fd /= fd2;

  // -------------------------------------------------------------------------
  // Test that / rules apply to % too, except that doubles are not allowed.
  // -------------------------------------------------------------------------

  sc = sc % sc2;
  uc = uc % uc2;
  bc = bc % bc2; // expected-error {{invalid operands}}

  sc = uc % sc2; // expected-error {{cannot convert}}
  sc = sc % bc2; // expected-error {{cannot convert}}
  uc = bc % uc2; // expected-error {{cannot convert}}

  sc %= sc2;
  uc %= uc2;
  bc %= bc2; // expected-error {{invalid operands}}

  sc %= uc2; // expected-error {{cannot convert}}
  uc %= bc2; // expected-error {{cannot convert}}
  bc %= sc2; // expected-error {{cannot convert}}

  ss %= ss2;
  us %= us2;
  bs %= bs2; // expected-error {{invalid operands}}

  si %= si2;
  ui %= ui2;
  bi %= bi2; // expected-error {{invalid operands}}

  sl %= sl2;
  ul %= ul2;
  bl %= bl2; // expected-error {{invalid operands}}

  slll %= slll2;
  ulll %= ulll2;
  blll %= blll2; // expected-error {{invalid operands}}

  fd %= fd2; // expected-error {{invalid operands}}

  // -------------------------------------------------------------------------
  // Test &.
  // -------------------------------------------------------------------------

  sc = sc & sc2;
  sc = sc & uc2; // expected-error {{cannot convert}}
  sc = uc & sc2; // expected-error {{cannot convert}}
  sc = sc & bc2;
  sc = bc & sc2;

  uc = uc & uc2;
  uc = sc & uc2; // expected-error {{cannot convert}}
  uc = uc & sc2; // expected-error {{cannot convert}}
  uc = bc & uc2;
  uc = uc & bc2;

  bc = bc & bc2;
  bc = bc & uc2; // expected-error {{incompatible type}}
  bc = uc & bc2; // expected-error {{incompatible type}}
  bc = bc & sc2; // expected-error {{incompatible type}}
  bc = sc & bc2; // expected-error {{incompatible type}}

  fd = fd & fd2; // expected-error {{invalid operands}}
  fd = bl & fd2; // expected-error {{invalid operands}}
  fd = fd & bl2; // expected-error {{invalid operands}}
  fd = fd & sl2; // expected-error {{invalid operands}}
  fd = fd & ul2; // expected-error {{invalid operands}}

  sc &= sc2;
  sc &= uc2; // expected-error {{cannot convert}}
  sc &= bc2;

  uc &= uc2;
  uc &= sc2; // expected-error {{cannot convert}}
  uc &= bc2;

  bc &= bc2;
  bc &= sc2; // expected-error {{cannot convert}}
  bc &= uc2; // expected-error {{cannot convert}}

  sc &= ss2; // expected-error {{cannot convert}}
  sc &= si2; // expected-error {{cannot convert}}
  sc &= sl2; // expected-error {{cannot convert}}
  sc &= fd2; // expected-error {{invalid operands}}

  us &= bc2; // expected-error {{cannot convert}}
  ui &= bc2; // expected-error {{cannot convert}}
  ul &= bc2; // expected-error {{cannot convert}}
  fd &= bc2; // expected-error {{invalid operands}}

  ss &= ss2;
  us &= us2;
  bs &= bs2;

  si &= si2;
  ui &= ui2;
  bi &= bi2;

  sl &= sl2;
  ul &= ul2;
  bl &= bl2;

  slll &= slll2;
  ulll &= ulll2;
  blll &= blll2;

  // -------------------------------------------------------------------------
  // Test that & rules apply to | too.
  // -------------------------------------------------------------------------

  sc = sc | sc2;
  sc = sc | uc2; // expected-error {{cannot convert}}
  sc = sc | bc2;

  uc = uc | uc2;
  uc = sc | uc2; // expected-error {{cannot convert}}
  uc = bc | uc2;

  bc = bc | bc2;
  bc = uc | bc2; // expected-error {{incompatible type}}
  bc = bc | sc2; // expected-error {{incompatible type}}

  fd = fd | fd2; // expected-error {{invalid operands}}
  fd = bl | fd2; // expected-error {{invalid operands}}

  ss |= ss2;
  us |= us2;
  bs |= bs2;

  si |= si2;
  ui |= ui2;
  bi |= bi2;

  sl |= sl2;
  ul |= ul2;
  bl |= bl2;

  slll |= slll2;
  ulll |= ulll2;
  blll |= blll2;

  fd |= bl2; // expected-error {{invalid operands}}
  fd |= fd2; // expected-error {{invalid operands}}

  // -------------------------------------------------------------------------
  // Test that & rules apply to ^ too.
  // -------------------------------------------------------------------------

  sc = sc ^ sc2;
  sc = sc ^ uc2; // expected-error {{cannot convert}}
  sc = sc ^ bc2;

  uc = uc ^ uc2;
  uc = sc ^ uc2; // expected-error {{cannot convert}}
  uc = bc ^ uc2;

  bc = bc ^ bc2;
  bc = uc ^ bc2; // expected-error {{incompatible type}}
  bc = bc ^ sc2; // expected-error {{incompatible type}}

  fd = fd ^ fd2; // expected-error {{invalid operands}}
  fd = bl ^ fd2; // expected-error {{invalid operands}}

  ss ^= ss2;
  us ^= us2;
  bs ^= bs2;

  si ^= si2;
  ui ^= ui2;
  bi ^= bi2;

  sl ^= sl2;
  ul ^= ul2;
  bl ^= bl2;

  slll ^= slll2;
  ulll ^= ulll2;
  blll ^= blll2;

  fd ^= bl2; // expected-error {{invalid operands}}
  fd ^= fd2; // expected-error {{invalid operands}}

  // -------------------------------------------------------------------------
  // Test <<.
  // -------------------------------------------------------------------------

  sc = sc << sc2;
  sc = sc << uc2;
  sc = uc << sc2; // expected-error {{incompatible type}}
  sc = sc << bc2; // expected-error {{invalid operands}}
  sc = bc << sc2; // expected-error {{invalid operands}}

  uc = uc << uc2;
  uc = sc << uc2; // expected-error {{assigning to}}
  uc = uc << sc2;
  uc = bc << uc2; // expected-error {{invalid operands}}
  uc = uc << bc2; // expected-error {{invalid operands}}

  bc = bc << bc2; // expected-error {{invalid operands}}
  bc = bc << uc2; // expected-error {{invalid operands}}
  bc = uc << bc2; // expected-error {{invalid operands}}
  bc = bc << sc2; // expected-error {{invalid operands}}
  bc = sc << bc2; // expected-error {{invalid operands}}

  sc = sc << 1;
  sc = sc << 1.0f; // expected-error {{integer is required}}
  sc = sc << sc_scalar;
  sc = sc << uc_scalar;
  sc = sc << ss_scalar;
  sc = sc << us_scalar;
  sc = sc << si_scalar;
  sc = sc << ui_scalar;
  sc = sc << sl_scalar;
  sc = sc << ul_scalar;
  sc = sc_scalar << sc; // expected-error {{first operand is not a vector}}
  sc = uc_scalar << sc; // expected-error {{first operand is not a vector}}
  uc = uc << sc_scalar;
  uc = uc << uc_scalar;
  uc = sc_scalar << uc; // expected-error {{first operand is not a vector}}
  uc = uc_scalar << uc; // expected-error {{first operand is not a vector}}

  ss = ss << ss2;
  ss = ss << ss_scalar;
  us = us << us2;
  us = us << us_scalar;
  bs = bs << bs2; // expected-error {{invalid operands}}

  si = si << si2;
  si = si << si_scalar;
  ui = ui << ui2;
  ui = ui << ui_scalar;
  bi = bi << bi2; // expected-error {{invalid operands}}

  sl = sl << sl2;
  sl = sl << sl_scalar;
  ul = ul << ul2;
  ul = ul << ul_scalar;
  bl = bl << bl2; // expected-error {{invalid operands}}

  slll = slll << slll2;
  slll = slll << slll_scalar;
  ulll = ulll << ulll2;
  ulll = ulll << ulll_scalar;
  blll = blll << blll2; // expected-error {{invalid operands}}

  fd = fd << fd2; // expected-error {{integer is required}}
  fd = fd << ul2; // expected-error {{integer is required}}
  fd = sl << fd2; // expected-error {{integer is required}}

  sc <<= sc2;
  sc <<= uc2;
  sc <<= bc2; // expected-error {{invalid operands}}
  sc <<= sc_scalar;

  uc <<= uc2;
  uc <<= sc2;
  uc <<= bc2; // expected-error {{invalid operands}}
  uc <<= uc_scalar;

  bc <<= bc2; // expected-error {{invalid operands}}
  bc <<= sc2; // expected-error {{invalid operands}}
  bc <<= uc2; // expected-error {{invalid operands}}

  sc <<= ss2; // expected-error {{vector operands do not have the same number of elements}}
  sc <<= si2; // expected-error {{vector operands do not have the same number of elements}}
  sc <<= sl2; // expected-error {{vector operands do not have the same number of elements}}
  sc <<= fd2; // expected-error {{integer is required}}

  ss <<= ss2;
  ss <<= ss_scalar;
  us <<= us2;
  us <<= us_scalar;
  bs <<= bs2; // expected-error {{invalid operands}}

  si <<= si2;
  si <<= si_scalar;
  ui <<= ui2;
  ui <<= ui_scalar;
  bi <<= bi2; // expected-error {{invalid operands}}

  sl <<= sl2;
  sl <<= sl_scalar;
  ul <<= ul2;
  ul <<= ul_scalar;
  bl <<= bl2; // expected-error {{invalid operands}}

  slll <<= slll2;
  slll <<= slll_scalar;
  ulll <<= ulll2;
  ulll <<= ulll_scalar;
  blll <<= blll2; // expected-error {{invalid operands}}

  fd <<= fd2; // expected-error {{integer is required}}

  // -------------------------------------------------------------------------
  // Test >>.
  // -------------------------------------------------------------------------

  sc = sc >> sc2;
  sc = sc >> uc2;
  sc = uc >> sc2; // expected-error {{incompatible type}}
  sc = sc >> bc2; // expected-error {{invalid operands}}
  sc = bc >> sc2; // expected-error {{invalid operands}}

  uc = uc >> uc2;
  uc = sc >> uc2; // expected-error {{assigning to}}
  uc = uc >> sc2;
  uc = bc >> uc2; // expected-error {{invalid operands}}
  uc = uc >> bc2; // expected-error {{invalid operands}}

  bc = bc >> bc2; // expected-error {{invalid operands}}
  bc = bc >> uc2; // expected-error {{invalid operands}}
  bc = uc >> bc2; // expected-error {{invalid operands}}
  bc = bc >> sc2; // expected-error {{invalid operands}}
  bc = sc >> bc2; // expected-error {{invalid operands}}

  sc = sc >> 1;
  sc = sc >> 1.0f; // expected-error {{integer is required}}
  sc = sc >> sc_scalar;
  sc = sc >> uc_scalar;
  sc = sc >> ss_scalar;
  sc = sc >> us_scalar;
  sc = sc >> si_scalar;
  sc = sc >> ui_scalar;
  sc = sc >> sl_scalar;
  sc = sc >> ul_scalar;
  sc = sc_scalar >> sc; // expected-error {{first operand is not a vector}}
  sc = uc_scalar >> sc; // expected-error {{first operand is not a vector}}
  uc = uc >> sc_scalar;
  uc = uc >> uc_scalar;
  uc = sc_scalar >> uc; // expected-error {{first operand is not a vector}}
  uc = uc_scalar >> uc; // expected-error {{first operand is not a vector}}

  ss = ss >> ss2;
  ss = ss >> ss_scalar;
  us = us >> us2;
  us = us >> us_scalar;
  bs = bs >> bs2; // expected-error {{invalid operands}}

  si = si >> si2;
  si = si >> si_scalar;
  ui = ui >> ui2;
  ui = ui >> ui_scalar;
  bi = bi >> bi2; // expected-error {{invalid operands}}

  sl = sl >> sl2;
  sl = sl >> sl_scalar;
  ul = ul >> ul2;
  ul = ul >> ul_scalar;
  bl = bl >> bl2; // expected-error {{invalid operands}}

  slll = slll >> slll2;
  slll = slll >> slll_scalar;
  ulll = ulll >> ulll2;
  ulll = ulll >> ulll_scalar;
  blll = blll >> blll2; // expected-error {{invalid operands}}

  fd = fd >> fd2; // expected-error {{integer is required}}
  fd = fd >> ul2; // expected-error {{integer is required}}
  fd = sl >> fd2; // expected-error {{integer is required}}

  sc >>= sc2;
  sc >>= uc2;
  sc >>= bc2; // expected-error {{invalid operands}}
  sc >>= sc_scalar;

  uc >>= uc2;
  uc >>= sc2;
  uc >>= bc2; // expected-error {{invalid operands}}
  uc >>= uc_scalar;

  bc >>= bc2; // expected-error {{invalid operands}}
  bc >>= sc2; // expected-error {{invalid operands}}
  bc >>= uc2; // expected-error {{invalid operands}}

  sc >>= ss2; // expected-error {{vector operands do not have the same number of elements}}
  sc >>= si2; // expected-error {{vector operands do not have the same number of elements}}
  sc >>= sl2; // expected-error {{vector operands do not have the same number of elements}}
  sc >>= fd2; // expected-error {{integer is required}}

  ss >>= ss2;
  ss >>= ss_scalar;
  us >>= us2;
  us >>= us_scalar;
  bs >>= bs2; // expected-error {{invalid operands}}

  si >>= si2;
  si >>= si_scalar;
  ui >>= ui2;
  ui >>= ui_scalar;
  bi >>= bi2; // expected-error {{invalid operands}}

  sl >>= sl2;
  sl >>= sl_scalar;
  ul >>= ul2;
  ul >>= ul_scalar;
  bl >>= bl2; // expected-error {{invalid operands}}

  slll >>= slll2;
  slll >>= slll_scalar;
  ulll >>= ulll2;
  ulll >>= ulll_scalar;
  blll >>= blll2; // expected-error {{invalid operands}}

  fd >>= fd2; // expected-error {{integer is required}}

  // -------------------------------------------------------------------------
  // Test ==.
  // -------------------------------------------------------------------------

  (void)(sc == sc2);
  (void)(uc == uc2);
  (void)(bc == bc2);

  (void)(sc == uc); // expected-error {{cannot convert}}
  (void)(sc == bc);

  (void)(uc == sc); // expected-error {{cannot convert}}
  (void)(uc == bc);

  (void)(bc == sc);
  (void)(bc == uc);

  (void)(ss == ss2);
  (void)(us == us2);
  (void)(bs == bs2);

  (void)(si == si2);
  (void)(ui == ui2);
  (void)(bi == bi2);

  (void)(sl == sl2);
  (void)(ul == ul2);
  (void)(bl == bl2);
  (void)(fd == fd2);

  (void)(slll == slll2);
  (void)(ulll == ulll2);
  (void)(blll == blll2);

  (void)(fd == ul); // expected-error {{cannot convert}}
  (void)(ul == fd); // expected-error {{cannot convert}}

  // -------------------------------------------------------------------------
  // Test that == rules apply to != too.
  // -------------------------------------------------------------------------

  (void)(sc != sc2);
  (void)(uc != uc2);
  (void)(bc != bc2);

  (void)(sc != uc); // expected-error {{cannot convert}}
  (void)(sc != bc);

  (void)(ss != ss2);
  (void)(us != us2);
  (void)(bs != bs2);

  (void)(si != si2);
  (void)(ui != ui2);
  (void)(bi != bi2);

  (void)(sl != sl2);
  (void)(ul != ul2);
  (void)(bl != bl2);
  (void)(fd != fd2);

  (void)(slll != slll2);
  (void)(ulll != ulll2);
  (void)(blll != blll2);

  // -------------------------------------------------------------------------
  // Test that == rules apply to <= too.
  // -------------------------------------------------------------------------

  (void)(sc <= sc2);
  (void)(uc <= uc2);
  (void)(bc <= bc2);

  (void)(sc <= uc); // expected-error {{cannot convert}}
  (void)(sc <= bc);

  (void)(ss <= ss2);
  (void)(us <= us2);
  (void)(bs <= bs2);

  (void)(si <= si2);
  (void)(ui <= ui2);
  (void)(bi <= bi2);

  (void)(sl <= sl2);
  (void)(ul <= ul2);
  (void)(bl <= bl2);
  (void)(fd <= fd2);

  (void)(slll <= slll2);
  (void)(ulll <= ulll2);
  (void)(blll <= blll2);

  // -------------------------------------------------------------------------
  // Test that == rules apply to >= too.
  // -------------------------------------------------------------------------

  (void)(sc >= sc2);
  (void)(uc >= uc2);
  (void)(bc >= bc2);

  (void)(sc >= uc); // expected-error {{cannot convert}}
  (void)(sc >= bc);

  (void)(ss >= ss2);
  (void)(us >= us2);
  (void)(bs >= bs2);

  (void)(si >= si2);
  (void)(ui >= ui2);
  (void)(bi >= bi2);

  (void)(sl >= sl2);
  (void)(ul >= ul2);
  (void)(bl >= bl2);
  (void)(fd >= fd2);

  (void)(slll >= slll2);
  (void)(ulll >= ulll2);
  (void)(blll >= blll2);

  // -------------------------------------------------------------------------
  // Test that == rules apply to < too.
  // -------------------------------------------------------------------------

  (void)(sc < sc2);
  (void)(uc < uc2);
  (void)(bc < bc2);

  (void)(sc < uc); // expected-error {{cannot convert}}
  (void)(sc < bc);

  (void)(ss < ss2);
  (void)(us < us2);
  (void)(bs < bs2);

  (void)(si < si2);
  (void)(ui < ui2);
  (void)(bi < bi2);

  (void)(sl < sl2);
  (void)(ul < ul2);
  (void)(bl < bl2);
  (void)(fd < fd2);

  (void)(slll < slll2);
  (void)(ulll < ulll2);
  (void)(blll < blll2);

  // -------------------------------------------------------------------------
  // Test that == rules apply to > too.
  // -------------------------------------------------------------------------

  (void)(sc > sc2);
  (void)(uc > uc2);
  (void)(bc > bc2);

  (void)(sc > uc); // expected-error {{cannot convert}}
  (void)(sc > bc);

  (void)(ss > ss2);
  (void)(us > us2);
  (void)(bs > bs2);

  (void)(si > si2);
  (void)(ui > ui2);
  (void)(bi > bi2);

  (void)(sl > sl2);
  (void)(ul > ul2);
  (void)(bl > bl2);
  (void)(fd > fd2);

  (void)(slll > slll2);
  (void)(ulll > ulll2);
  (void)(blll > blll2);
}
