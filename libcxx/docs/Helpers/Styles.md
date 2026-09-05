---
substitutions:
  Complete: '{complete}`Complete`'
  In Progress: '{inprogress}`In Progress`'
  Not Started: '{notstarted}`Not Started`'
  Nothing To Do: '{nothingtodo}`Nothing To Do`'
  Partial: '{partial}`Partial`'
  Review: '{inreview}`Review`'
  hellip: |-
    ```{eval-rst}
    .. unicode:: U+2026
    ```
  sect: |-
    ```{eval-rst}
    .. unicode:: U+00A7
    ```
---

```{raw} html
<style type="text/css">
  td { text-align: left; }
  .notstarted { opacity: 60%; }
  .nothingtodo {
      background-color: #99FF99;
      font-style: italic;
   }
  .inprogress {
      background-color: #FFFF99;
      font-style: italic;
   }
  .inreview { background-color: #FFFF99; }
  .partial {
      background-color: #2CCCFF;
      font-style: italic;
   }
  .complete { background-color: #99FF99; }
</style>
```

```{eval-rst}
.. role:: notstarted
```

```{eval-rst}
.. role:: nothingtodo
```

```{eval-rst}
.. role:: inprogress
```

```{eval-rst}
.. role:: inreview
```

```{eval-rst}
.. role:: partial
```

```{eval-rst}
.. role:: complete
```

