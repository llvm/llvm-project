// RUN: c-index-test -test-load-source-reparse 1 local %s | FileCheck %s

// See PR 21254. We had too few bits to encode command IDs so if you created
// enough of them the ID codes would wrap around. This test creates commands up
// to an ID of 258. Ideally we should check for large numbers, but that would
// require a test source file which is megabytes in size. This is the test case
// from the PR.

/**
@s
@tr
@y
@tt
@tg
@alu
@U
@I
@r
@t0
@t1
@ur
@S
@E
@pb
@f
@pe
@lue
@re
@oa
@l
@x
@R
@ute
@am
@ei
@oun
@ou
@nlnl
@ien
@fr
@en
@teq
@le
@L
@os
@A
@ro
@o
@ho
@ca
@Tie
@tl
@g
@hr
@et
@fro
@ast
@ae
@nNnN
@pc
@tae
@ws
@ia
@N
@lc
@psg
@ta
@t2
@D
@str
@ra
@t3
@t
@xt
@eN
@fe
@rU
@ar
@eD
@iEx
@se
@st1
@rr
@ime
@ft
@lm
@wD
@wne
@h
@otn
@use
@roi
@ldc
@ln
@d
@ee
@ep
@us
@ut
@u
@n
@Nme
@min
@ma
@pct
@hd
@be
@It
@idz
@cm
@ua
@fs
@Al
@axn
@rt
@to
@isj
@fo
@i
@an
@de
@tel
@ndnd
@dict
@Lo
@ilil
@tle
@axt
@ba
@ust
@ac
@tpe
@tpl
@ctG
@ru
@m
@tG
@itit
@rh
@G
@rpc
@el
@er
@w
@eo
@tx
@oo
@dD
@dD
*/
void f();

// CHECK:  CommentAST=[
// CHECK:    (CXComment_FullComment
// CHECK:       (CXComment_Paragraph
// CHECK:         (CXComment_InlineCommand CommandName=[s] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[tr] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[y] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[tt] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[tg] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[alu] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[U] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[I] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[r] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[t0] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[t1] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ur] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[S] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[E] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[pb] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[f] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[pe] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[lue] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[re] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[oa] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[l] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[x] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[R] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ute] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[am] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ei] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[oun] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ou] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[nlnl] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ien] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[fr] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[en] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[teq] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[le] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[L] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[os] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[A] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ro] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[o] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ho] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ca] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[Tie] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[tl] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[g] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[hr] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[et] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[fro] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ast] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ae] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[nNnN] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[pc] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[tae] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ws] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ia] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[N] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[lc] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[psg] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ta] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[t2] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[D] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[str] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ra] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[t3] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[t] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[xt] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[eN] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[fe] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[rU] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ar] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[eD] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[iEx] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[se] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[st1] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[rr] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ime] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ft] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[lm] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[wD] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[wne] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[h] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[otn] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[use] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[roi] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ldc] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ln] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[d] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ee] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ep] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[us] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ut] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[u] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[n] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[Nme] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[min] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ma] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[pct] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[hd] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[be] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[It] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[idz] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[cm] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ua] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[fs] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[Al] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[axn] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[rt] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[to] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[isj] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[fo] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[i] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[an] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[de] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[tel] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ndnd] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[dict] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[Lo] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ilil] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[tle] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[axt] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ba] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ust] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ac] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[tpe] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[tpl] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ctG] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[ru] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[m] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[tG] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[itit] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[rh] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[G] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[rpc] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[el] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[er] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[w] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[eo] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[tx] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[oo] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[dD] RenderNormal HasTrailingNewline)
// CHECK:         (CXComment_InlineCommand CommandName=[dD] RenderNormal)))]
