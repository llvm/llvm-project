MODULE modfile73a
PUBLIC re_alloc, defaults
integersp  
integerselected_real_kind0
integer:: i8b = selected_int_kind(8)
interface
   subroutine alloc_error_report_interf(str,code)
   end  
   subroutine alloc_memory_event_interf(bytes,name)
   end  
end interface
procedure()alloc_error_report  
procedure()alloc_memory_event  
  interface de_alloc
  end interface
  charactercharacter, DEFAULT_ROUTINE  
  type allocDefaults
    logical copy
    logical shrink
    integer imin
    characterroutine
  end type 
  type(allocDefaults)DEFAULT
  integer IERR
  logical ASSOCIATED_ARRAY, NEEDS_ALLOC, NEEDS_COPY, NEEDS_DEALLOC
CONTAINS
  subroutine set_alloc_event_handler(func)
  end  
  subroutine set_alloc_error_handler(func)
  end  
  subroutine dummy_alloc_memory_event(bytes,name)
  end  
  subroutine dummy_alloc_error_report(name,code)
  end  
SUBROUTINE alloc_default( old, new, restore,          routine, copy, shrink, imin )
END  
SUBROUTINE realloc_i1( array, i1min, i1max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_i2( array, i1min,i1max, i2min,i2max,       name, routine, copy, shrink )
END  
SUBROUTINE realloc_i3( array, i1min,i1max, i2min,i2max, i3min,i3max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_i4( array, i1min,i1max, i2min,i2max, i3min,i3max, i4min,i4max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_i5( array, i1min,i1max, i2min,i2max, i3min,i3max, i4min, i4max, i5min, i5max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_E1( array, i1min, i1max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_r1( array, i1min, i1max,        name, routine, copy, shrink )
END  
SUBROUTINE realloc_r2( array, i1min,i1max, i2min,i2max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_r3( array, i1min,i1max, i2min,i2max, i3min,i3max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_r4( array, i1min,i1max, i2min,i2max, i3min,i3max, i4min,i4max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_r5( array, i1min,i1max, i2min,i2max, i3min,i3max, i4min, i4max, i5min, i5max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_d1( array, i1min, i1max,        name, routine, copy, shrink )
END  
SUBROUTINE realloc_d2( array, i1min,i1max, i2min,i2max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_d3( array, i1min,i1max, i2min,i2max, i3min,i3max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_d4( array, i1min,i1max, i2min,i2max, i3min,i3max, i4min,i4max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_d5( array, i1min,i1max, i2min,i2max, i3min,i3max, i4min,i4max, i5min,i5max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_c1( array, i1min, i1max,        name, routine, copy, shrink )
END  
SUBROUTINE realloc_c2( array, i1min,i1max, i2min,i2max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_c3( array, i1min,i1max, i2min,i2max, i3min,i3max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_c4( array, i1min,i1max, i2min,i2max, i3min,i3max, i4min,i4max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_c5( array, i1min,i1max, i2min,i2max, i3min,i3max, i4min, i4max, i5min, i5max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_z1( array, i1min, i1max,        name, routine, copy, shrink )
END  
SUBROUTINE realloc_z2( array, i1min,i1max, i2min,i2max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_z3( array, i1min,i1max, i2min,i2max, i3min,i3max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_z4( array, i1min,i1max, i2min,i2max, i3min,i3max, i4min,i4max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_z5( array, i1min,i1max, i2min,i2max, i3min,i3max, i4min, i4max, i5min, i5max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_l1( array, i1min,i1max,  name, routine, copy, shrink )
END  
SUBROUTINE realloc_l2( array, i1min,i1max, i2min,i2max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_l3( array, i1min,i1max, i2min,i2max, i3min,i3max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_l4( array, i1min,i1max, i2min,i2max, i3min,i3max, i4min,i4max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_l5( array, i1min,i1max, i2min,i2max, i3min,i3max, i4min, i4max, i5min, i5max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_i1s( array, i1max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_i2s( array, i1max, i2max,  name, routine, copy, shrink )
END  
SUBROUTINE realloc_i3s( array, i1max, i2max, i3max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_r1s( array, i1max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_r2s( array, i1max, i2max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_r3s( array, i1max, i2max, i3max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_r4s( array, i1max, i2max, i3max, i4max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_d1s( array, i1max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_d3s( array, i1max, i2max, i3max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_d4s( array, i1max, i2max, i3max, i4max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_l1s( array, i1max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_l2s( array, i1max, i2max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_l3s( array, i1max, i2max, i3max, name, routine, copy, shrink )
END  
SUBROUTINE realloc_s1( array, i1min, i1max, name, routine, copy, shrink )
END  
SUBROUTINE dealloc_i1( array, name, routine )
END  
SUBROUTINE dealloc_i2( array, name, routine )
END  
SUBROUTINE dealloc_i3( array, name, routine )
END  
SUBROUTINE dealloc_i4( array, name, routine )
  END  
SUBROUTINE dealloc_i5( array, name, routine )
  END  
SUBROUTINE dealloc_E1( array, name, routine )
END  
SUBROUTINE dealloc_r1( array, name, routine )
END  
SUBROUTINE dealloc_r2( array, name, routine )
END  
SUBROUTINE dealloc_r3( array, name, routine )
END  
SUBROUTINE dealloc_r4( array, name, routine )
END  
SUBROUTINE dealloc_r5( array, name, routine )
END  
SUBROUTINE dealloc_d1( array, name, routine )
END  
SUBROUTINE dealloc_d2( array, name, routine )
END  
SUBROUTINE dealloc_d3( array, name, routine )
END  
SUBROUTINE dealloc_d4( array, name, routine )
END  
SUBROUTINE dealloc_d5( array, name, routine )
END  
SUBROUTINE dealloc_c1( array, name, routine )
END  
SUBROUTINE dealloc_c2( array, name, routine )
END  
SUBROUTINE dealloc_c3( array, name, routine )
END  
SUBROUTINE dealloc_c4( array, name, routine )
END  
SUBROUTINE dealloc_c5( array, name, routine )
  END  
SUBROUTINE dealloc_z1( array, name, routine )
END  
SUBROUTINE dealloc_z2( array, name, routine )
END  
SUBROUTINE dealloc_z3( array, name, routine )
END  
SUBROUTINE dealloc_z4( array, name, routine )
END  
SUBROUTINE dealloc_z5( array, name, routine )
  END  
SUBROUTINE dealloc_l1( array, name, routine )
END  
SUBROUTINE dealloc_l2( array, name, routine )
END  
SUBROUTINE dealloc_l3( array, name, routine )
END  
SUBROUTINE dealloc_l4( array, name, routine )
  END  
SUBROUTINE dealloc_l5( array, name, routine )
  END  
SUBROUTINE dealloc_s1( array, name, routine )
END  
SUBROUTINE options( final_bounds, common_bounds, old_bounds, new_bounds, copy, shrink )
END  
SUBROUTINE alloc_err( ierr, name, routine, bounds )
END  
SUBROUTINE alloc_count( delta_size, type, name, routine )
END  
END
