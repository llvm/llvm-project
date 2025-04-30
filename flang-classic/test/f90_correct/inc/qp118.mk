#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test qtolog  ########


qp118: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp118.f08 fcheck.$(OBJX)
	-$(RM) qp118.$(EXESUFFIX) qp118.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp118.f08 -o qp118.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp118.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp118.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp118
	qp118.$(EXESUFFIX)

verify: ;

qp118.run: run

