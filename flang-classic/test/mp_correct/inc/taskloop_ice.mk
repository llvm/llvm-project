# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

build: taskloop_ice.$(OBJX)

run:
	@echo ------------ executing test $@
	-$(RUN2) ./taskloop_ice.$(EXESUFFIX) $(LOG)

verify: ;

taskloop_ice.$(OBJX): $(SRC)/taskloop_ice.f90 check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/taskloop_ice.f90
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) taskloop_ice.$(OBJX) check.$(OBJX) $(LIBS) -o taskloop_ice.$(EXESUFFIX)

