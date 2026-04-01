'use strict';

module.exports = {
  math_dialect : $ => choice(
                   // operation ::= `math.absf` $operand (`fastmath` ``
                   // $fastmath^)?
                   //               attr-dict `:` type($result)
                   // operation ::= `math.atan` $operand (`fastmath` ``
                   // $fastmath^)?
                   //               attr-dict `:` type($result)
                   // operation ::= `math.cbrt` $operand (`fastmath` ``
                   // $fastmath^)?
                   //               attr-dict `:` type($result)
                   // operation ::= `math.ceil` $operand (`fastmath` ``
                   // $fastmath^)?
                   //               attr-dict `:` type($result)
                   // operation ::= `math.cos` $operand (`fastmath` ``
                   // $fastmath^)?
                   //               attr-dict `:` type($result)
                   // operation ::= `math.erf` $operand (`fastmath` ``
                   // $fastmath^)?
                   //               attr-dict `:` type($result)
                   // operation ::= `math.exp` $operand (`fastmath` ``
                   // $fastmath^)?
                   //               attr-dict `:` type($result)
                   // operation ::= `math.exp2` $operand (`fastmath` ``
                   // $fastmath^)?
                   //               attr-dict `:` type($result)
                   // operation ::= `math.expm1` $operand (`fastmath` ``
                   // $fastmath^)?
                   //               attr-dict `:` type($result)
                   // operation ::= `math.floor` $operand (`fastmath` ``
                   // $fastmath^)?
                   //               attr-dict `:` type($result)
                   // operation ::= `math.log` $operand (`fastmath` ``
                   // $fastmath^)?
                   //               attr-dict `:` type($result)
                   // operation ::= `math.log10` $operand (`fastmath` ``
                   // $fastmath^)?
                   //               attr-dict `:` type($result)
                   // operation ::= `math.log1p` $operand (`fastmath` ``
                   // $fastmath^)?
                   //               attr-dict `:` type($result)
                   // operation ::= `math.log2` $operand (`fastmath` ``
                   // $fastmath^)?
                   //               attr-dict `:` type($result)
                   // operation ::= `math.round` $operand (`fastmath` ``
                   // $fastmath^)?
                   //               attr-dict `:` type($result)
                   // operation ::= `math.roundeven` $operand (`fastmath` ``
                   // $fastmath^)?
                   //               attr-dict `:` type($result)
                   // operation ::= `math.rsqrt` $operand (`fastmath` ``
                   // $fastmath^)?
                   //               attr-dict `:` type($result)
                   // operation ::= `math.sin` $operand (`fastmath` ``
                   // $fastmath^)?
                   //               attr-dict `:` type($result)
                   // operation ::= `math.sqrt` $operand (`fastmath` ``
                   // $fastmath^)?
                   //               attr-dict `:` type($result)
                   // operation ::= `math.tan` $operand (`fastmath` ``
                   // $fastmath^)?
                   //               attr-dict `:` type($result)
                   // operation ::= `math.tanh` $operand (`fastmath` ``
                   // $fastmath^)?
                   //               attr-dict `:` type($result)
                   // operation ::= `math.trunc` $operand (`fastmath` ``
                   // $fastmath^)?
                   //               attr-dict `:` type($result)
                   seq(choice('math.absf', 'math.atan', 'math.cbrt',
                              'math.ceil', 'math.cos', 'math.erf', 'math.exp',
                              'math.exp2', 'math.expm1', 'math.floor',
                              'math.log', 'math.log10', 'math.log1p',
                              'math.log2', 'math.round', 'math.roundeven',
                              'math.rsqrt', 'math.sin', 'math.sqrt', 'math.tan',
                              'math.tanh', 'math.trunc'),
                       field('operand', $.value_use),
                       field('fastmath', optional($.fastmath_attr)),
                       field('attributes', optional($.attribute)),
                       field('return', $._type_annotation)),

                   // operation ::= `math.absi` $operand attr-dict `:`
                   // type($result) operation ::= `math.ctlz` $operand attr-dict
                   // `:` type($result) operation ::= `math.cttz` $operand
                   // attr-dict `:` type($result) operation ::= `math.ctpop`
                   // $operand attr-dict `:` type($result)
                   seq(choice('math.absi', 'math.ctlz', 'math.cttz',
                              'math.ctpop'),
                       field('operand', $.value_use),
                       field('attributes', optional($.attribute)),
                       field('return', $._type_annotation)),

                   // operation ::= `math.atan2` $lhs `,` $rhs (`fastmath` ``
                   // $fastmath^)?
                   //               attr-dict `:` type($result)
                   // operation ::= `math.copysign` $lhs `,` $rhs (`fastmath` ``
                   // $fastmath^)?
                   //               attr-dict `:` type($result)
                   // operation ::= `math.fpowi` $lhs `,` $rhs (`fastmath` ``
                   // $fastmath^)?
                   //               attr-dict `:` type($lhs) `,` type($rhs)
                   // operation ::= `math.powf` $lhs `,` $rhs (`fastmath` ``
                   // $fastmath^)?
                   //               attr-dict `:` type($result)
                   seq(choice('math.atan2', 'math.copysign', 'math.fpowi',
                              'math.powf'),
                       field('lhs', $.value_use), ',',
                       field('rhs', $.value_use),
                       field('fastmath', optional($.fastmath_attr)),
                       field('attributes', optional($.attribute)),
                       field('return', $._type_annotation)),

                   // operation ::= `math.ipowi` $lhs `,` $rhs attr-dict `:`
                   // type($result)
                   seq('math.ipowi', field('lhs', $.value_use), ',',
                       field('rhs', $.value_use),
                       field('attributes', optional($.attribute)),
                       field('return', $._type_annotation)),

                   // operation ::= `math.fma` $a `,` $b `,` $c (`fastmath` ``
                   // $fastmath^)?
                   //               attr-dict `:` type($result)
                   seq('math.fma', field('a', $.value_use), ',',
                       field('b', $.value_use), ',', field('c', $.value_use),
                       field('fastmath', optional($.fastmath_attr)),
                       field('attributes', optional($.attribute)),
                       field('return', $._type_annotation)))
}
