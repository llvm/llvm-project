#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test log8toq ########


qp78: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp78.f08 fcheck.$(OBJX)
	-$(RM) qp78.$(EXESUFFIX) qp78.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp78.f08 -o qp78.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp78.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp78.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp78
	qp78.$(EXESUFFIX)

verify: ;

qp78.run: run

