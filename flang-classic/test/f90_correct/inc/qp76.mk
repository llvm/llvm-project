#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test int8toq ########


qp76: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp76.f08 fcheck.$(OBJX)
	-$(RM) qp76.$(EXESUFFIX) qp76.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp76.f08 -o qp76.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp76.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp76.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp76
	qp76.$(EXESUFFIX)

verify: ;

qp76.run: run

