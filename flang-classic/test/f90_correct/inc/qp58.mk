#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test qtobint  ########


qp58: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp58.f08 fcheck.$(OBJX)
	-$(RM) qp58.$(EXESUFFIX) qp58.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp58.f08 -o qp58.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp58.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp58.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp58
	qp58.$(EXESUFFIX)

verify: ;

qp58.run: run

