# Design note: `[[clang::reinitializes]]` vs the predicate role verbs

## The question

`reinitializes` already exists and is described as returning an object "to a known
state." Our `disengaged` role makes an object empty. Are we writing an attribute
that does the *opposite* of an existing one — or duplicating it?

**No.** They live on two independent axes. They frequently co-occur on the same
method (`reset`/`clear`) but answer different questions for different checks.

## Two axes

1. **Determinacy** — what `reinitializes` addresses: *is the object in a known
   state, or is it moved-from / indeterminate?* Moves you `indeterminate →
   determinate`. Consumed by `bugprone-use-after-move` / the static analyzer.
   Says **nothing** about full vs empty — the doc is explicit: "a known state,
   independent of the previous state."

2. **Predicate** — what `engaged`/`disengaged` address: *is the state bit true or
   false (full vs empty)?* Consumed by the unchecked-optional-access predicate
   model.

`disengaged` is therefore **not** the negation of `reinitializes`. The negation
of `disengaged` is `engaged` — that opposite pair is inside our own vocabulary,
by design (the polarity axis). `reinitializes` is not on that axis at all.

## Why they co-occur without conflicting

`clear()` legitimately carries both, because it does two true things at once:

- makes the object **determinate** (`reinitializes`) → use-after-move stops warning;
- makes the predicate **false / empty** (`disengaged`) → the model knows a later
  `front()`/`value()` is unsafe.

A moved-from container you then `clear()` is now *both* determinate *and* empty.
Valid-and-empty is a normal state; empty ≠ invalid.

|                | `reinitializes` (real, today)        | `disengaged` (proposed role)         |
|----------------|--------------------------------------|--------------------------------------|
| Question       | still moved-from / indeterminate?    | is the predicate bit true/false?     |
| Axis           | determinacy                          | predicate (full/empty)               |
| Consumer       | `bugprone-use-after-move` / analyzer | unchecked-optional-access model       |
| After `clear()`| object is determinate again          | predicate is now false (empty)       |

## Why we can't just reuse `reinitializes`

The fair version of "do we even need a new one?" — yes:

- It says "a **known** state," **not "empty."** A type's known state could be
  non-empty (e.g. a container that resets to a sentinel element), so you cannot
  reliably infer "predicate false" from it.
- It is a **lone verb** — there is no `reinitializes`-family with
  engaged/test/assume counterparts. It cannot express "requires the bit," "sets
  the bit true," or "queries the bit." The predicate model needs a whole
  vocabulary; `reinitializes` is one point on a different map.

## Framing for the RFC

`reinitializes` is a **precedent that supports the proposal**, not a competitor:
clang already ships a narrow, per-method "returns object to a defined state"
attribute consumed by exactly one analysis. That is the shape we are proposing,
for a different analysis. State the roles as **complementary** to `reinitializes`
(they will frequently sit together on `reset`/`clear`), never as a replacement or
an inversion of it.

See also `architecture.md` (pipeline + validation levels) and
`why_class_comparison.md` (identity vs generic vocabulary).
