#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test stoq ########


qp94: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp94.f08 fcheck.$(OBJX)
	-$(RM) qp94.$(EXESUFFIX) qp94.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp94.f08 -o qp94.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp94.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp94.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp94
	qp94.$(EXESUFFIX)

verify: ;

qp94.run: run

