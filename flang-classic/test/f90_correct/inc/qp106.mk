#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test dtoq  ########


qp106: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp106.f08 fcheck.$(OBJX)
	-$(RM) qp106.$(EXESUFFIX) qp106.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp106.f08 -o qp106.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp106.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp106.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp106
	qp106.$(EXESUFFIX)

verify: ;

qp106.run: run

