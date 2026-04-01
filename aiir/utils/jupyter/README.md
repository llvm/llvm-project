A Jupyter kernel for aiir (aiir-opt)

This is purely for experimentation. This kernel uses the reproducer runner
conventions to run passes.

To install:

    python3 -m aiir_opt_kernel.install

To use it, run one of:

```shell
    jupyter notebook
    # In the notebook interface, select AiirOpt from the 'New' menu
    jupyter console --kernel aiir
```

`aiir-opt` is expected to be either in the `PATH` or `AIIR_OPT_EXECUTABLE` is
used to point to the executable directly.
