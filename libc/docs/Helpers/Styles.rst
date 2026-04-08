.. ---------------------------------------------------------------------------
   Shared substitution definitions for implementation-status badges.
   This file is excluded from the toctree (via exclude_patterns in conf.py).
   Include it in any RST page that needs status badges:

       .. include:: /Helpers/Styles.rst   (adjust path as needed)

   Usage::

       |Complete|     — feature/function is fully implemented
       |Partial|      — partial implementation (some variants/platforms missing)
       |InProgress|   — implementation underway, not yet merged
       |NotStarted|   — no implementation exists yet
       |GPUOnly|      — implemented on GPU targets only
       |LinuxOnly|    — implemented on Linux targets only
   ---------------------------------------------------------------------------

.. |Complete| raw:: html

   <span class="badge badge-complete">Complete</span>

.. |Partial| raw:: html

   <span class="badge badge-partial">Partial</span>

.. |InProgress| raw:: html

   <span class="badge badge-inprogress">In Progress</span>

.. |NotStarted| raw:: html

   <span class="badge badge-notstarted">Not Started</span>

.. |GPUOnly| raw:: html

   <span class="badge badge-gpuonly">GPU Only</span>

.. |LinuxOnly| raw:: html

   <span class="badge badge-linuxonly">Linux Only</span>
