  .globl  _start
_start:
  .functype _start () -> ()
  end_function

  .globl     someByte
  .type      someByte,@object
  .section  .data.someByte,"",@
someByte:
  .int8 42
  .size someByte, 1
