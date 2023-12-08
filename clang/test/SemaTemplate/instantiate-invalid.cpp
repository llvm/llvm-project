// RUN: not %clang_cc1 -fsyntax-only %s
namespace PR6375 {
  template<class Conv> class rasterizer_sl_clip Conv::xi(x2), Conv::yi(y2));
namespace agg
{
	template<class Clip=rasterizer_sl_clip_int> class rasterizer_scanline_aa
	{
		template<class Scanline> bool sweep_scanline(Scanline& sl)
		{
			unsigned num_cells = m_outline.scanline_num_cells(m_scan_y);
			while(num_cells) { }
		}
	}
    class scanline_u8 {}
    template<class PixelFormat> class renderer_base { }
}
    template<class Rasterizer, class Scanline, class BaseRenderer, class ColorT>
    void render_scanlines_aa_solid(Rasterizer& ras, Scanline& sl, BaseRenderer& ren, const ColorT& color)
    {
            while(ras.sweep_scanline(sl))
            {
        }
    };
namespace agg
{
    struct rgba8
    {
    };
    template<class Rasterizer, class Scanline, class Renderer, class Ctrl>
    void render_ctrl(Rasterizer& ras, Scanline& sl, Renderer& r, Ctrl& c)
    {
        unsigned i;
            render_scanlines_aa_solid(ras, sl, r, c.color(i));
        }
    template<class ColorT> class rbox_ctrl : public rbox_ctrl_impl
    {
        const ColorT& color(unsigned i) const { return *m_colors[i]; }
    }
class the_application : public agg::platform_support
{
    agg::rbox_ctrl<agg::rgba8> m_polygons;
    virtual void on_init()
    {
        typedef agg::renderer_base<pixfmt_type> base_ren_type;
        base_ren_type ren_base(pf);
        agg::scanline_u8 sl;
        agg::rasterizer_scanline_aa<> ras;
        agg::render_ctrl(ras, sl, ren_base, m_polygons);
    }
};
}
}
