# Error recovery (2022-05-07)

Problem: we have two fairly generic cases of recovery bounded within a range:
 - sequences: `int x; this is absolute garbage; x++;`
 - brackets: `void foo() { this is garbage too }`

So far, we've thought of these as different, and had precise ideas about
brackets ("lazy parsing") and vague ones about sequences.
Con we unify them instead?

In both cases we want to recognize the bounds of the garbage item based on
basic token-level features of the surrounding code, and avoid any interference
with the surrounding code.

## Brackets

Consider a rule like `compound-stmt := { stmt-seq }`.

The desired recovery is:
- if we fail at `{ . stmt-seq }`
- ... and we can find for the matching `}`
- then consume up to that token as an opaque broken `stmt-seq`
- ... and advance state to `{ stmt-seq . }`

We can annotate the rule to describe this: `{ stmt-seq [recovery] }`.
We can generalize as `{ stmt-seq [recovery=rbrace] }`, allowing for different
**strategies** to find the resume point.

(It's OK to assume we're skipping over one RHS nonterminal, we can always
introduce a nonterminal for the bracket contents if necessary).

## Sequences

Can we apply the same technique to sequences?
Simplest case: delimited right-recursive sequence.

```
param-list := param
param-list := param , param-list
```

We need recovery in **both** rules.
`param` in the first rule matches the *last* param in a list,
in the second rule it matches all others. We want to recover in any position.

If we want to be able to recovery `int x int y` as two parameters, then we
should extract a `param-and-comma` rule that recovery could operate on.

### Last vs not-last elements

Sequences will usually have two rules with recovery, we discussed:
 - how to pick the correct one to recover with
 - in a left-recursive rule they correspond to last & not-last elements
 - the recovery strategy knows whether we're recoverying last or not-last
 - we could have the strategy return (pos, symbol parsed), and artificially
   require distinct symbols (e.g. `stmt` vs `last_stmt`).
 - we can rewrite left-recursion in the grammar as right-recursion

However, on reflection I think we can simply allow recovery according to both
rules. The "wrong" recovery will produce a parse head that dies.

## How recovery fits into GLR

Recovery should kick in at the point where we would otherwise abandon all
variants of an high-level parse.

e.g. Suppose we're parsing `static_cast<foo bar baz>(1)` and are up to `bar`.
Our GSS looks something like:

```
     "the static cast's type starts at foo"
---> {expr := static_cast < . type > ( expr )}
         |     "foo... is a class name"
         +---- {type := identifier .}
         |     "foo... is a template ID"
         +---- {type := identifier . < template-arg-list >}
```

Since `foo bar baz` isn't a valid class name or template ID, both active heads
will soon die, as will the parent GSS Node - the latter should trigger recovery.

- we need a refcount in GSS nodes so we can recognize never-reduced node death
- when a node dies, we look up its recovery actions (symbol, strategy).
  These are the union of the recovery actions for each item.
  They can be stored in the action table.
  Here: `actions[State, death] = Recovery(type, matching-angle-bracket)`
- we try each strategy: feeding in the start position = token of the dying node
  (`foo`) getting out the end position (`>`).
- We form an opaque forest node with the correct symbol (`type`) spanning
  [start, end)
- We create a GSS node to represent the state after recovery.
  The new state is found in the Goto table in the usual way.

```
     "the static cast's type starts at foo"
---> {expr := static_cast < . type > ( expr )}
         |     "`foo bar baz` is an unparseable type"
         +---- {expr := static_cast < type . > (expr)}
```

## Which recovery heads to activate

We probably shouldn't *always* create active recovery heads when a recoverable
node dies (and thus keep it alive).
By design GLR creates multiple speculative parse heads and lets incorrect heads
disappear.

Concretely, the expression `(int *)(x)` is a valid cast, we probably shouldn't
also parse it as a call whose callee is a broken expr.

The simplest solution is to only create recovery heads if there are no normal
heads remaining, i.e. if parsing is completely stuck. This is vulnerable if the
"wrong" parse makes slightly more progress than the "right" parse which has
better error recovery.

A sophisticated variant might record recovery opportunities and pick the one
with the rightmost *endpoint* when the last parse head dies.

We should consider whether including every recovery in the parse forest might
work after all - this would let disambiguation choose "broken" but likely parses
over "valid" but unlikely ones.


