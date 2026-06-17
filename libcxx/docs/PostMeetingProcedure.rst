.. _PostMeetingProcedure:

===========================
Post-meeting procedure
===========================

The C++ standards committee (WG21) meets several times a year. Each plenary
session adopts new papers and Library Working Group (LWG) issues that libc++
needs to track. This page describes the procedure that libc++ developers must
follow after each WG21 plenary meeting to keep the conformance trackers in
sync with what the committee voted in.

The tracker files
=================

For each version of the C++ standard, libc++ maintains some CSV files under
``libcxx/docs/Status/``:

* ``Cxx<NN>Papers.csv`` — every WG21 paper with library impact that targets
  this standard version.
* ``Cxx<NN>Issues.csv`` — every LWG issue that targets this standard version.

Each row in the CSVs corresponds to one Github tracking issue on ``llvm/llvm-project``.
Those tracking issues are also linked to the `libc++ Standards Conformance
<https://github.com/orgs/llvm/projects/31>`__ project. Together, the CSVs, the tracking
issues and the project board are kept in sync by the ``libcxx/utils/conformance`` script.

When tracking new items after a plenary vote, the CSV files should be updated first, and
then the Github issues should be created from the CSV files using the script.

Deciding what plenary motions to track
======================================

After each meeting, look at the meeting's straw polls page (requires being a member
of WG21) and decide what needs to be added to the trackers. Generally speaking, we
want to track every plenary-approved LWG paper or LWG issue. We also want to track
CWG papers and issues that have library wording or impact (but we don't track other
CWG papers and issues).

Also note that LWG and CWG issues are respectively bundled as a single motion/paper
on the straw polls page. The papers in these motions contain the actual issues that
we should be tracking.

To confirm that an issue or paper was approved in plenary, ``https://wg21.link/<PAPER>/status``
can be used. That will link to the Github issue tracking the paper in WG21's system,
where papers approved in plenary have the ``plenary-approved`` label.

Updating the CSV files
======================

For each new paper or issue to track, add a row following the convention used in existing
files. If a paper was voted as a Defect Report, mention it in the notes. The syntax of the
file can be validated with::

   libcxx/utils/conformance csv validate libcxx/docs/Status/Cxx<NN>Papers.csv \
                                         libcxx/docs/Status/Cxx<NN>Issues.csv

Commit the CSV changes and related updates as a first PR. The Github tracking issues are
created in a separate step.

Link stray Github issues
========================

People sometimes create Github issues to track standard papers outside of the workflow
described here. While that should be discouraged as only plenary-voted papers should be
tracked and this workflow should be used, issues created outside of this workflow should
still be linked to prevent duplicates and confusion. This can be done with::

   libcxx/utils/conformance github find-unlinked --labels wg21-paper --labels lwg-issue

This will find existing issues with the given labels that are not linked to the Github project
tracking conformance. They can then be linked manually.

Create the Github tracking issues
==================================

Once the CSV files are committed and any stray issues have been linked, the remaining missing
Github issues can be created using ``libcxx/utils/conformance``. The script lists every issue
it would create from the CSV (skipping rows that are already tracked) and asks for a confirmation
before creating them. The issue title, body and labels are all populated automatically, and they
are appropriately linked to the libc++ conformance project. Run it once per CSV::

   libcxx/utils/conformance github create libcxx/docs/Status/Cxx<NN>Papers.csv \
      --labels=wg21-paper --labels=c++<NN>

   libcxx/utils/conformance github create libcxx/docs/Status/Cxx<NN>Issues.csv \
      --labels=lwg-issue --labels=c++<NN>

The CSV ``Notes`` column is written to the issue body between ``BEGIN-RST-NOTES``/``END-RST-NOTES``
markers so that ``csv synchronize`` can round-trip it back into the CSV.

Once the issues have been created, populate the ``GitHub issue`` column of each CSV row::

   libcxx/utils/conformance csv synchronize libcxx/docs/Status/Cxx<NN>Papers.csv \
      -o libcxx/docs/Status/Cxx<NN>Papers.csv

   libcxx/utils/conformance csv synchronize libcxx/docs/Status/Cxx<NN>Issues.csv \
      -o libcxx/docs/Status/Cxx<NN>Issues.csv

This can then be committed as a follow-up PR.
