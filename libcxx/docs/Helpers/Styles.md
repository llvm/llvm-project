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
  .experimental { background-color: #FFB000 }

  /* Furo's dark theme uses light text, so the pastel backgrounds above are
     unreadable there. Furo selects dark mode either explicitly or, for the
     "auto" theme, from the browser's color scheme preference. */
  body[data-theme="dark"] .nothingtodo,
  body[data-theme="dark"] .complete { background-color: #1f5c2e; }
  body[data-theme="dark"] .inprogress,
  body[data-theme="dark"] .inreview { background-color: #6b6414; }
  body[data-theme="dark"] .partial { background-color: #155a78; }
  body[data-theme="dark"] .experimental { background-color: #86580a; }
  @media (prefers-color-scheme: dark) {
    body:not([data-theme="light"]) .nothingtodo,
    body:not([data-theme="light"]) .complete { background-color: #1f5c2e; }
    body:not([data-theme="light"]) .inprogress,
    body:not([data-theme="light"]) .inreview { background-color: #6b6414; }
    body:not([data-theme="light"]) .partial { background-color: #155a78; }
    body:not([data-theme="light"]) .experimental { background-color: #86580a; }
  }
</style>
```

:::{role} notstarted
:::

:::{role} nothingtodo
:::

:::{role} inprogress
:::

:::{role} inreview
:::

:::{role} partial
:::

:::{role} complete
:::

:::{role} experimental
:::
