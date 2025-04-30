#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test qtobint  ########


qp111: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp111.f08 fcheck.$(OBJX)
	-$(RM) qp111.$(EXESUFFIX) qp111.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp111.f08 -o qp111.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp111.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp111.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp111
	qp111.$(EXESUFFIX)

verify: ;

qp111.run: run

