#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test logival8toq  ########


qp57: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp57.f08 fcheck.$(OBJX)
	-$(RM) qp57.$(EXESUFFIX) qp57.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp57.f08 -o qp57.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp57.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp57.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp57
	qp57.$(EXESUFFIX)

verify: ;

qp57.run: run

