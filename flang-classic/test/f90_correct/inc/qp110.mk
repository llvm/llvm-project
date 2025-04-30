#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test logival8toq  ########


qp110: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp110.f08 fcheck.$(OBJX)
	-$(RM) qp110.$(EXESUFFIX) qp110.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp110.f08 -o qp110.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp110.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp110.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp110
	qp110.$(EXESUFFIX)

verify: ;

qp110.run: run

