# Size distributions for memory functions under specific workloads

This folder contains a set of files that are included from `libc/benchmarks/MemorySizeDistributions.cpp`.

Offloading this data to individual files helps
 - C++ editors (large arrays are usually not well handled by editors),
 - and allows processing data by other tools to perform analysis or graph rendering.

 ## Format

Most filenames are of the form `{MemoryFunctionName}{WorkloadID}.csv`. They contain a single line of comma separated real values representing the probability that a particular size occurs. e.g.
 - `"0,1"` indicates that only the size `1` occurs,
 - `"0.5,0.5"` indicates sizes `0` and `1` occur with the same frequency.

 These files usually contains sizes from `0` to `4096` inclusive. To save on space trailing zeros are discarded.

 ## Workloads

As identified in the [automemcpy](https://research.google/pubs/pub50338/) paper:
  - `GoogleA` <-> `service 4`
  - `GoogleB` <-> `database 1`
  - `GoogleD` <-> `storage`
  - `GoogleL` <-> `logging`
  - `GoogleM` <-> `service 2`
  - `GoogleQ` <-> `database 2`
  - `GoogleS` <-> `database 3`
  - `GoogleU` <-> `service 3`
  - `GoogleW` <-> `service 1`

`Uniform384To4096` is an additional synthetic workload that simply returns a uniform repartition of the sizes from `384` to `4096` inclusive.

## Note

Except for `GoogleD`, all distributions are gathered over one week worth of data.